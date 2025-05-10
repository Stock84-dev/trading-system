// use cuda_std::GpuFloat;

// Use delta average when there isn't enough instruction level parallelism. Otherwise use KahanSum
// as it will have lower overall latency because it contains no divisions.
pub struct DeltaAvg {
    avg: f32,
}

impl DeltaAvg {
    #[inline(always)]
    pub fn new() -> DeltaAvg {
        DeltaAvg { avg: 0.0 }
    }

    #[inline(always)]
    pub fn update(&mut self, x: f32, i: u32) {
        avg(&mut self.avg, x, i);
    }

    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.avg
    }
}

#[inline(always)]
pub fn avg(avg: &mut f32, x: f32, i: u32) {
    let recp_n = 1.0 / (i + 1) as f32;
    *avg = recp_n.mul_add(x - *avg, *avg);
}

pub struct KahanSum {
    sum: f32,
    c: f32,
}

impl KahanSum {
    #[inline(always)]
    pub fn new() -> KahanSum {
        KahanSum { sum: 0.0, c: 0.0 }
    }

    #[inline(always)]
    pub fn add(&mut self, x: f32) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline(always)]
    pub fn sum(&self) -> f32 {
        self.sum
    }
}

pub struct KahanSum64 {
    sum: f64,
    c: f64,
}

impl KahanSum64 {
    #[inline(always)]
    pub fn new() -> KahanSum64 {
        KahanSum64 { sum: 0.0, c: 0.0 }
    }

    #[inline(always)]
    pub fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline(always)]
    pub fn sum(&self) -> f64 {
        self.sum
    }
}

pub struct CorrelationState {
    pub avg_x: f32,
    pub avg_y: f32,
    pub var_x: f32,
    pub var_y: f32,
    pub cov_xy: f32,
}

impl CorrelationState {
    #[inline(always)]
    pub fn new() -> CorrelationState {
        CorrelationState {
            avg_x: 0.0,
            avg_y: 0.0,
            var_x: 0.0,
            var_y: 0.0,
            cov_xy: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, x: f32, y: f32, i: u32) {
        let recp_n = 1.0 / (i + 1) as f32;
        let delta_x = x - self.avg_x;
        let delta_y = y - self.avg_y;
        self.avg_x = recp_n.mul_add(delta_x, self.avg_x);
        self.avg_y = recp_n.mul_add(delta_y, self.avg_y);
        let one_minus_recip_n = 1.0 - recp_n;

        self.var_x = one_minus_recip_n * recp_n.mul_add(delta_x * delta_x, self.var_x);
        self.var_y = one_minus_recip_n * recp_n.mul_add(delta_y * delta_y, self.var_y);
        self.cov_xy = one_minus_recip_n * recp_n.mul_add(delta_x * delta_y, self.cov_xy);
    }

    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.cov_xy / (self.var_x * self.var_y).sqrt()
    }
}

pub struct StdDev {
    pub avg_squared: DeltaAvg,
    pub avg: DeltaAvg,
}

impl StdDev {
    #[inline(always)]
    pub fn new() -> StdDev {
        StdDev {
            avg_squared: DeltaAvg::new(),
            avg: DeltaAvg::new(),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, x: f32, i: u32) {
        self.avg.update(x, i);
        self.avg_squared.update(x * x, i);
    }

    #[inline(always)]
    pub fn value(&self) -> f32 {
        (self.avg_squared.value() - self.avg.value() * self.avg.value()).sqrt()
    }
}

pub struct StdDev2 {
    pub sum_squared: KahanSum64,
    pub sum: KahanSum64,
}

impl StdDev2 {
    #[inline(always)]
    pub fn new() -> StdDev2 {
        StdDev2 {
            sum_squared: KahanSum64::new(),
            sum: KahanSum64::new(),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, x: f32) {
        let x = x as f64;
        self.sum.add(x);
        self.sum_squared.add(x * x);
    }

    #[inline(always)]
    pub fn value(&self, n_samples: usize) -> f32 {
        let n_samples = n_samples as f64;
        let mean = self.sum.sum() / n_samples;
        let sq_mean = self.sum_squared.sum() / n_samples;
        let variance = sq_mean - mean * mean;
        variance.sqrt() as f32
    }
}

