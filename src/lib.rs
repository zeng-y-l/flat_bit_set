#![doc = include_str!("../README.md")]
#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![deny(unsafe_code)]
#![warn(clippy::pedantic, missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::wildcard_imports)]

extern crate alloc;

mod inner;
#[cfg(test)]
mod test;

pub use inner::*;
