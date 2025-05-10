use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_attribute]
pub fn custom(_attributes: TokenStream, input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    use std::io::Read;
    // let mut file =
    // std::fs::File::open("/home/stock/ssd/projects/project-fusion/exturion/esl/src/
    // metrics_builder.rs").unwrap();
    let mut file = std::fs::File::open(
        "/home/stock/ssd/projects/project-fusion/exturion/exturion_strategies/src/\
         exturion_strategies.rs",
    )
    .unwrap();

    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    let now = std::time::Instant::now();
    let _file = syn::parse_file(&content).unwrap();
    eprintln!("{}us", now.elapsed().as_micros());
    quote! {
        #[attr1]
        #input
    }
    .into()
}

#[proc_macro_attribute]
pub fn attr1(_attributes: TokenStream, input: TokenStream) -> TokenStream {
    input
}
