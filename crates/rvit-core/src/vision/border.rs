pub trait BorderMode: Clone + Copy {}
#[derive(Debug, Clone, Copy, Default)]
pub struct Constant {}
impl BorderMode for Constant {}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClampToEdge {}
impl BorderMode for ClampToEdge {}

#[derive(Debug, Clone, Copy, Default)]
pub struct Mirror {}
impl BorderMode for Mirror {}
