pub trait InterpMode: Clone + Copy {}

#[derive(Default, Copy, Clone)]
pub struct Bilinear {}

impl InterpMode for Bilinear {}
