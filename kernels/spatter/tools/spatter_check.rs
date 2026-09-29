//! Run a fused Spatter ELF on Cyclotron and validate its output in GPU memory.

use cyclotron::base::mem::HasMemory;
use cyclotron::ui::{make_sim, read_toml, CyclotronArgs};
use std::path::PathBuf;

fn parse_hex(value: &str) -> usize {
    usize::from_str_radix(value.trim_start_matches("0x"), 16).expect("invalid hex address")
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() != 6 && (argv.len() != 7 || argv[6] != "--functional") {
        eprintln!("usage: spatter_check CONFIG ELF GUARD_BEFORE_HEX OUTPUT_ELEMENTS EXPECTED_DIGEST_HEX_OR_NONE [--functional]");
        std::process::exit(2);
    }
    let config_path = PathBuf::from(&argv[1]);
    let elf = PathBuf::from(&argv[2]);
    let guard_before = parse_hex(&argv[3]);
    let output_elements: usize = argv[4].parse().expect("invalid output element count");
    let output_addr = guard_before + 64;
    let output_bytes = output_elements.checked_mul(8).expect("output length overflow");
    let expected = if argv[5] == "none" { None } else { Some(parse_hex(&argv[5]) as u64) };

    let options = CyclotronArgs {
        config_path: config_path.clone(),
        binary_path: Some(elf),
        timing: argv.len() == 6,
        ..CyclotronArgs::default()
    };
    let toml_string = read_toml(config_path.as_path());
    let mut sim = make_sim(Some(&toml_string), &Some(options));
    if let Err(code) = sim.simulate() {
        eprintln!("Cyclotron failed or timed out with code {code}");
        std::process::exit(1);
    }

    let mem = sim.top.gmem.read().expect("GPU memory lock poisoned");
    let before = mem.read_impl(guard_before, 64).expect("guard before out of range");
    let after = mem.read_impl(output_addr + output_bytes, 64).expect("guard after out of range");
    let guards_intact = before.iter().all(|&byte| byte == 0)
        && after.iter().all(|&byte| byte == 0);
    let output = mem.read_impl(output_addr, output_bytes).expect("output out of range");
    let mut digest = 0xCBF29CE484222325u64;
    let mut nonzero_words = 0u64;
    for word in output.chunks_exact(4) {
        let value = u32::from_le_bytes(word.try_into().unwrap());
        if value != 0 {
            nonzero_words += 1;
        }
        digest ^= u64::from(value);
        digest = digest.wrapping_mul(0x100000001B3);
    }
    let digest_ok = expected.map_or(true, |value| value == digest);
    println!(
        "SPATTER_CHECK digest={digest:016x} expected={} guards_intact={guards_intact} nonzero_words={nonzero_words} output_elements={output_elements}",
        expected.map_or_else(|| "none".to_owned(), |value| format!("{value:016x}"))
    );
    if !guards_intact || !digest_ok || nonzero_words == 0 {
        std::process::exit(1);
    }
}
