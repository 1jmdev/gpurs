use gpurs::{gpu::bit, *};

#[gpu]
fn unpack_colors(packed: &[u32], channels: &mut [u32]) {
    let index = gpu::index();

    if index < packed.len() {
        let pixel = packed[index];
        let red = bit::extract_bits(pixel, 16, 8);
        let green = bit::extract_bits(pixel, 8, 8);
        let blue = bit::extract_bits(pixel, 0, 8);
        let swapped = bit::insert_bits(pixel, red, 0, 8);
        let swapped = bit::insert_bits(swapped, blue, 16, 8);
        let reversed = bit::permute_bytes(pixel, 0, 0x0123);
        let rotated = bit::funnel_shift_left(pixel, pixel, 4);

        channels[index] = red + green + blue + (swapped ^ reversed ^ rotated).count_ones();
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let data = (0..1024)
        .map(|value| value * 0x010203)
        .collect::<Vec<u32>>();

    let packed = gpu.upload(&data);
    let mut channels = gpu.zeros::<u32>(data.len());

    unpack_colors(&stream, &packed, &mut channels);

    println!("Channels: {:?}", &channels.download()[..8]);
}
