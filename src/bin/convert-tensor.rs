pub fn main() {
    let args: Vec<_> = std::env::args().collect();
    if !args.len() == 3 {
        eprintln!("usage: {} source.npz destination.ot", args[0]);
        std::process::exit(1);
    }

    let source_file = &args[1];
    let destination_file = &args[2];
    let tensors = tch::Tensor::read_npz(source_file).unwrap();
    tch::Tensor::save_multi(&tensors, destination_file).unwrap();
}
