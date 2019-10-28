#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

pub mod raw {
    #![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
    include!(concat!(env!("OUT_DIR"), "/petsc_raw.rs"));
}

include!(concat!(env!("OUT_DIR"), "/petsc.rs"));
