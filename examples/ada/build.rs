use gpurs_build::Embedding;

fn main() {
    gpurs_build::compile_with(&["sm_89"], Embedding::Ptx);
}
