use std::path::PathBuf;

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}

fn build_cuda() {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];

    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let cu_roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];

    let cu_roots = cu_roots.into_iter().map(Into::<PathBuf>::into);
    let cu_root = env_vars
        .chain(cu_roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .unwrap();

    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        cu_root.join("include").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cu_root.join("lib").display()
    );

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

    // Glob for all .cu files.
    // TODO: Will need to do this for .cuh files when there are any.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let kernel_paths: Vec<PathBuf> = glob::glob("src/**/*.cu")
        .unwrap()
        .map(|path| path.unwrap())
        .collect();

    // Check whether any of the cuda files have been updated.
    kernel_paths
        .iter()
        .for_each(|path| println!("cargo:rerun-if-changed={}", path.display()));

    // Generate .ptx files for all cuda files found.
    // TODO: Should state the SM version here - probably derived from nvidia-smi.
    let children = kernel_paths
        .iter()
        .map(|path| {
            std::process::Command::new("nvcc")
                .arg("--ptx")
                .args(["--default-stream", "per-thread"])
                .args(["--output-directory", &out_dir])
                .arg(path)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .expect("Unable to start nvcc. Ensure that the nvcc directory is on PATH.")
        })
        .collect::<Vec<_>>();

    // Wait for all nvcc processes to finish and output any failures.
    for (kernel_path, child) in kernel_paths.iter().zip(children.into_iter()) {
        let output = child
            .wait_with_output()
            .expect("Unable to start nvcc. Ensure that the nvcc directory is on PATH.");
        assert!(
            output.status.success(),
            "nvcc error while compiling {kernel_path:?}:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
