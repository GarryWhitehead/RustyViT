#[cfg(feature = "vulkan")]
use shaderc::ShaderKind;
#[cfg(feature = "vulkan")]
use std::{
    fs,
    path::{Path, PathBuf},
};

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
    #[cfg(feature = "vulkan")]
    build_vulkan();
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "vulkan")]
fn to_rust_type(ty: &str) -> &str {
    match ty {
        "uint8_t" => "u8",
        "uint16_t" => "u16",
        "float" => "f32",
        "double" => "f64",
        _ => panic!("Unknown type {ty}"),
    }
}

#[cfg(feature = "vulkan")]
fn build_vulkan() {
    let compiler = shaderc::Compiler::new().unwrap();

    // Note: vision and tensor shaders are compiled separately due mainly to different macro definitions.
    // Glob for vision glsl files.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let vision_glsl_paths: Vec<PathBuf> = glob::glob("shaders/vulkan/vision/*.glsl")
        .unwrap()
        .map(|path| path.unwrap())
        .collect();
    vision_glsl_paths
        .iter()
        .for_each(|path| println!("cargo:rerun-if-changed={}", path.display()));

    vision_glsl_paths.iter().for_each(|path| {
        ["uint8_t", "uint16_t", "float"].map(|ty| {
            let filename = Path::new(path).file_name().unwrap().to_str().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            options.add_macro_definition("PIXELTYPE", Some(ty));
            let glsl_src = fs::read_to_string(&path).unwrap();
            let artifact = compiler
                .compile_into_spirv(
                    &glsl_src,
                    ShaderKind::Compute,
                    filename,
                    "main",
                    Some(&options),
                )
                .unwrap();
            fs::write(
                format!(
                    "{}/{}_{}.spv",
                    out_dir,
                    to_rust_type(ty),
                    Path::new(filename).file_stem().unwrap().to_str().unwrap()
                ),
                artifact.as_binary_u8(),
            )
            .unwrap();
        });
    });

    // Tensor shaders.
    let tensor_glsl_paths: Vec<PathBuf> = glob::glob("shaders/vulkan/tensor/*.glsl")
        .unwrap()
        .map(|path| path.unwrap())
        .collect();
    tensor_glsl_paths
        .iter()
        .for_each(|path| println!("cargo:rerun-if-changed={}", path.display()));

    tensor_glsl_paths.iter().for_each(|path| {
        [("float", "float32_t")].map(|ty| {
            let filename = Path::new(path).file_name().unwrap().to_str().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            options.add_macro_definition("FLOAT_TYPE", Some(ty.0));
            options.add_macro_definition("COOP_FLOAT_TYPE", Some(ty.1));
            options.set_target_env(
                shaderc::TargetEnv::Vulkan,
                shaderc::EnvVersion::Vulkan1_3 as u32,
            );
            let glsl_src = fs::read_to_string(&path).unwrap();
            let artifact = compiler
                .compile_into_spirv(
                    &glsl_src,
                    ShaderKind::Compute,
                    filename,
                    "main",
                    Some(&options),
                )
                .unwrap();
            fs::write(
                format!(
                    "{}/{}_{}.spv",
                    out_dir,
                    to_rust_type(ty.0),
                    Path::new(filename).file_stem().unwrap().to_str().unwrap()
                ),
                artifact.as_binary_u8(),
            )
            .unwrap();
        });
    });
}
