pub fn div_up(a: u32, b: u32) -> u32 {
    assert!(b > 0);
    (a + b - 1) / b
}
