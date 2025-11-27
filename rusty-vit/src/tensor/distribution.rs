use crate::type_traits::FloatType;
use rand::distr::uniform::SampleUniform;
use rand_distr::{Distribution, StandardNormal};
use rand_distr::{Normal, Uniform};

#[derive(Copy, Clone, Debug)]
pub enum DistributionMethod {
    KaimingNormal,
    KaimingUniform,
    Normal,
    Uniform,
}

pub fn sample<T: FloatType + SampleUniform>(
    fan_in: usize,
    _fan_out: usize,
    buffer: &mut [T],
    method: DistributionMethod,
) where
    StandardNormal: Distribution<f64>,
{
    let mut rng = rand::rng();
    match method {
        DistributionMethod::KaimingUniform => {
            let s = (2.0 / fan_in as f64).sqrt();
            let u = Uniform::new(0.0f64, s).unwrap();
            buffer
                .iter_mut()
                .for_each(|x| *x = T::from(u.sample(&mut rng)).unwrap())
        }
        DistributionMethod::KaimingNormal => {
            let s = (2.0 / fan_in as f64).sqrt();
            let n: Normal<f64> = Normal::new(0.0f64, s).unwrap();
            buffer
                .iter_mut()
                .for_each(|x| *x = T::from(n.sample(&mut rng)).unwrap());
        }
        DistributionMethod::Uniform => {
            let u = Uniform::new(-0.01f64, 0.1f64).unwrap();
            buffer
                .iter_mut()
                .for_each(|x| *x = T::from(u.sample(&mut rng)).unwrap());
        }
        DistributionMethod::Normal => {
            let n = Normal::new(0.0f64, 0.1f64).unwrap();
            buffer
                .iter_mut()
                .for_each(|x| *x = T::from(n.sample(&mut rng)).unwrap());
        }
    };
}
