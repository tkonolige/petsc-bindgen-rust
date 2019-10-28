use petsc_rs::*;

#[test]
fn initialize() {
    unsafe {
        let universe = mpi::initialize().unwrap();
        raw::PetscInitializeNoArguments();
        let world = universe.world();
        Mat::new(world);
        raw::PetscFinalize();
    }
}
