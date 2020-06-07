extern crate bindgen;
extern crate pkg_config;
extern crate quote;
extern crate syn;

use multimap::MultiMap;
use quote::{format_ident, quote};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::process::*;
use syn::fold::*;
use syn::Item::*;

fn format_rust_expression(value: &str) -> Cow<'_, str> {
    if let Ok(mut proc) = Command::new("rustfmt")
        .arg("--emit=stdout")
        .arg("--edition=2018")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        {
            let stdin = proc.stdin.as_mut().unwrap();
            stdin.write_all(value.as_bytes()).unwrap();
        }
        if let Ok(output) = proc.wait_with_output() {
            if output.status.success() {
                return std::str::from_utf8(&output.stdout)
                    .unwrap()
                    .to_owned()
                    .into();
            }
        }
    }
    Cow::Borrowed(value)
}

/// Executes the supplied console command, returning the `stdout` output if the
/// command was successfully executed.
fn run_command(command: &str, arguments: &[&str]) -> Option<String> {
    macro_rules! warn {
        ($error:expr) => {
            println!(
                "cargo:warning=couldn't execute `{} {}` ({})",
                command,
                arguments.join(" "),
                $error,
            );
        };
    }

    let output = match Command::new(command).args(arguments).output() {
        Ok(output) => output,
        Err(error) => {
            warn!(format!("error: {}", error));
            return None;
        }
    };

    if !output.status.success() {
        warn!(format!("exit code: {}", output.status));
        return None;
    }

    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn to_string<T>(a: &T) -> String
where
    T: quote::ToTokens,
    T: Clone,
{
    format!("{}", a.clone().into_token_stream())
}

struct QualifyType<'a> {
    raw_names: &'a HashSet<String>,
}
impl<'a> Fold for QualifyType<'a> {
    fn fold_type_path(&mut self, pth: syn::TypePath) -> syn::TypePath {
        let mut path = fold_path(self, pth.path);
        if path.segments.len() == 1
            && self
                .raw_names
                .contains(&to_string(path.segments.last().unwrap()))
        {
            path.segments.insert(
                0,
                syn::PathSegment {
                    ident: format_ident!("raw"),
                    arguments: syn::PathArguments::None,
                },
            )
        }
        syn::TypePath {
            qself: pth.qself.map(|q| fold_qself(self, q)),
            path: path,
        }
    }
}

// Mapping from petsc types to rust equivalents
fn replace_types(
    raw_names: &HashSet<String>,
    pats: Vec<syn::PatType>,
) -> (
    Vec<syn::export::TokenStream2>,
    Vec<syn::export::TokenStream2>,
    Vec<syn::export::TokenStream2>,
    Vec<syn::export::TokenStream2>,
) {
    let mut type_map = HashMap::new();
    // TODO: add PetscBool
    type_map.insert(
        "MPI_Comm",
        (quote! {AsRaw<Raw = MPI_Comm>}, format_ident!("as_raw")),
    );

    let mut args = Vec::new();
    let mut generics = Vec::new();
    let mut generic_names = Vec::new();
    let mut calls = Vec::new();

    for (i, pat) in pats.iter().enumerate() {
        match type_map.get(to_string(&(*pat.ty)).as_str()) {
            Some((rty, rfn)) => {
                // add a generic type for MPI_Comm
                // i.e. fn new<T_0>() -> (arg_1: T_0) where T_0: AsRaw<Raw = MPI_Comm>
                let patpat = &pat.pat;
                let generic_type = format_ident!("T_{}", i);
                generics.push(quote! { #generic_type : #rty });
                args.push(quote! { #patpat : #generic_type });
                calls.push(quote! { #patpat.#rfn() });
                generic_names.push(quote! { #generic_type });
            }
            None => {
                let patpat = &pat.pat;
                let ty = QualifyType { raw_names }.fold_type((*pat.ty).clone());
                args.push(quote! { #patpat: #ty });
                calls.push(quote! { #patpat });
            }
        }
    }

    (args, generic_names, generics, calls)
}

/// Convert from petsc constructor naming convention to rust. bool indicates if function is a
/// constructor or not.
fn replace_name(class: &syn::Ident, x: &str) -> (syn::Ident, bool) {
    if x == format!("{}Create", class) {
        (format_ident!("new"), true)
    } else if x == format!("{}Clone", class) {
        (format_ident!("clone"), true)
    } else {
        let class_name = format!("{}", class);
        (format_ident!("{}", x[class_name.len()..]), false)
    }
}

/// Find the longest match between class name and function name.
/// Example: given classes [Mat, MatNest], MatNestGetSubMat -> MatNest
fn longest_match(names: &HashSet<String>, name: &str) -> Option<String> {
    let m = names
        .iter()
        .map(|n| {
            if name.starts_with(n) {
                (n, n.len())
            } else {
                (n, 0)
            }
        })
        .max_by_key(|x| x.1);
    match m {
        Some((_, 0)) | None => None,
        Some((class, _)) => Some(class.to_string()),
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=PETSC_DIR");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    let petsc_dir = env::var("PETSC_DIR").map(|x| Path::new(&x).to_path_buf());
    let petsc_full_dir = petsc_dir.clone().map(|dir| match env::var("PETSC_ARCH") {
        Ok(arch) => dir.join(arch),
        Err(_) => dir.to_path_buf(),
    });

    // Add petsc path to pkg config path
    if let Ok(dir) = &petsc_full_dir {
        let pkgconfig_dir = dir.join("lib/pkgconfig");
        if let Some(path) = env::var_os("PKG_CONFIG_PATH") {
            let mut paths = env::split_paths(&path).collect::<Vec<_>>();
            paths.push(pkgconfig_dir);
            let new_path = env::join_paths(paths).unwrap();
            env::set_var("PKG_CONFIG_PATH", &new_path);
        } else {
            env::set_var("PKG_CONFIG_PATH", pkgconfig_dir);
        }
    }

    let lib = pkg_config::Config::new()
        .atleast_version("3.10")
        .probe("petsc")
        .unwrap();

    let out_dir = env::var("OUT_DIR").unwrap();

    for path in lib.link_paths {
        println!("cargo:rustc-link-search={}", path.to_string_lossy());
    }
    for lib in lib.libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut builder = bindgen::builder()
    .header("wrapper.h")
    .whitelist_function("[A-Z][a-zA-Z]*")
    .whitelist_type("[A-Z][a-zA-Z]*")
    .opaque_type("FILE")
    .blacklist_type("MPI\\w*")
    .raw_line("pub use mpi::ffi::*;")
    .default_enum_style(bindgen::EnumVariation::Rust{non_exhaustive:false}) // Need unstable rust for non exhaustive enums
    ;

    // LLVM can't find the correct system headers on MacOS because they aren't in /usr/include.
    if cfg!(target_os = "macos") {
        if let Some(output) = run_command("xcrun", &["--show-sdk-path"]) {
            let directory = Path::new(output.lines().next().unwrap()).to_path_buf();
            builder = builder.clang_arg(format!("-isysroot{}", directory.to_string_lossy()));
        }
    }

    for path in lib.include_paths {
        println!("cargo:rerun-if-changed={}", path.to_string_lossy());
        builder = builder.clang_arg(format!("-I{}", path.to_string_lossy()));
    }
    // Needed on MacOS for mpi
    builder = builder.clang_arg("-I/usr/local/include");

    let raw_bindings = Path::new(&out_dir).join("petsc_raw.rs");
    builder
        .generate()
        .expect("Cannot generate petsc bindings")
        .write_to_file(&raw_bindings)
        .expect("Cannot write petsc bindings");

    // parse the bindings generated by bindgen
    let mut file = File::open(&raw_bindings).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    let raw = syn::parse_file(&content).expect("Could not read generated bindings");

    // Collect the names of all petsc datastructures
    let raw_structs = raw
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Enum(eitem) => Some(to_string(&eitem.ident)),
            syn::Item::Struct(sitem) => Some(to_string(&sitem.ident)),
            syn::Item::Type(titem) => Some(to_string(&titem.ident)), // e.g. typedef *_p_SNESS SNES
            _ => None,
        })
        .collect::<HashSet<_>>();

    // Collect all function signatures
    let signatures = raw
        .items
        .iter()
        .filter_map(|item| match item {
            ForeignMod(it) => match &it.items[0] {
                syn::ForeignItem::Fn(fnitem) => {
                    // Ignore functions with no arguments
                    // TODO: we need this for PetscInitializeNoArguments
                    if fnitem.sig.inputs.len() <= 0 {
                        None
                    } else {
                        let fnname = fnitem.sig.ident.to_string();
                        // which petsc class does this belong to?
                        // TODO: instead of using a longest string match on the function name, we
                        // should look at the first argument.
                        let class = longest_match(&raw_structs, &fnname);
                        class.map(|cname| (cname, (fnname, fnitem.sig.clone())))
                    }
                }
                _ => None,
            },
            _ => None,
        })
        .collect::<MultiMap<String, (String, syn::Signature)>>();

    // create safe rust functions for each petsc function
    let defs = raw_structs.clone().into_iter().filter_map(|class| {
        let class_ident = format_ident!("{}", &class);

        // petsc function signatures generally look like PetscFunction(input1, input2, output)
        // right now the ones that don't follow this convention aren't handled
        match signatures.get_vec(&class.to_string()) {
            None => None,
            Some(sigs) => {
                let fns = sigs.iter().map( | (fnname, sig) | {
                    // last "input" is really our output
                    let (fnname, isconstructor) = replace_name(&class_ident, fnname);
                    let dropnum = if isconstructor { 0 } else { 1 }; // drop first argument as it becomes self
                    let fnargs = sig.inputs.iter().skip(dropnum).take(sig.inputs.len()-1).map(|x| match x {
                        syn::FnArg::Typed(pattype) => pattype.clone(),
                        syn::FnArg::Receiver(_) => panic!("Did not expect a self in the type signature"),
                    }).collect::<Vec<_>>();

                    let (mut args, generic_names, generics, mut calls) = replace_types(&raw_structs, fnargs);
                    if isconstructor {
                        calls.push(quote!{&mut raw as *mut raw::#class_ident});
                    } else {
                        let mut tmp = calls.clone();
                        calls = Vec::new();
                        calls.push(quote!{self.raw});
                        calls.append(&mut tmp);

                        // TODO: determine if this should take a mutable reference or not
                        let mut tmp = args.clone();
                        args = Vec::new();
                        args.push(quote!{&mut self});
                        args.append(&mut tmp);
                    }

                    let raw_fn = &sig.ident;
                    if isconstructor {
                        quote!{
                            pub fn #fnname<#(#generic_names),*>(#(#args),*) -> Self where #(#generics),* {
                                unsafe {
                                    let mut raw = MaybeUninit::uninit().as_mut_ptr();
                                    raw::#raw_fn(#(#calls),* , );
                                    #class_ident {raw: raw}
                                }
                            }
                        }
                    } else {
                        quote!{
                            pub fn #fnname<#(#generic_names),*>(#(#args),*) -> () where #(#generics),* {
                                unsafe {
                                    raw::#raw_fn(#(#calls),* , );
                                }
                            }
                        }
                    }
                }).collect::<Vec<_>>();

                Some(to_string(&quote!{
                    pub struct #class_ident {
                        raw: raw::#class_ident
                    }
                    impl #class_ident {
                        #(#fns)*
                    }
                }))
            }
        }
    }).collect::<Vec<_>>().join("\n");
    let mut f = File::create(Path::new(&out_dir).join("petsc.rs")).unwrap();
    let wrapped = format!(
        "use std::mem::MaybeUninit;\nuse mpi::ffi::*;\nuse mpi::raw::*;\n{}",
        defs
    );
    f.write(format_rust_expression(&wrapped).as_bytes())
        .unwrap();
}
