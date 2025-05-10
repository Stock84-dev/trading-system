use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// # Safety
/// The future must not return `Poll::Pending` when polled and interact with Context.
#[inline(always)]
pub unsafe fn block_on_unchecked<F: Future>(mut future: F) -> F::Output {
    fn noop_waker_fn(_data: *const ()) {}
    fn noop_raw_waker() -> RawWaker {
        RawWaker::new(
            core::ptr::null(),
            &RawWakerVTable::new(clone_raw_waker, noop_waker_fn, noop_waker_fn, noop_waker_fn),
        )
    }
    fn clone_raw_waker(_data: *const ()) -> RawWaker {
        noop_raw_waker()
    }

    unsafe {
        let mut future = Pin::new_unchecked(&mut future);
        let waker = Waker::from_raw(noop_raw_waker());
        let mut context = Context::from_waker(&waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(val) => val,
            Poll::Pending => {
                core::hint::unreachable_unchecked();
            },
        }
    }
}
