## Plotting tips:
Plotting storage consists of multiple accounts and streams. Accounts are used to store multiple instances of plot data, must be known at compile time (`DebugContext` takes it). 
A stream is a vec of plot primitives (line, point, vertical bar, candle, string). It must be monotonic with respect to time.
```Rust
CTX.debug(0, 0, timestamp_ns).add((
    lines!(n_positions, 0, 2),
    texts!((balance, s), 0, 1),
));
```
The above expression produces a stream for account 0 (first parameter) and writes it to stream 0 (second parameter). Every `CTX.debug()` expression must have a unique stream id.
The purpose of a stream is to separate out data that is generated conditionally and make it easier to plot at different areas of code. A stream can contain a maximum of 8 primitives. Create another stream if that's not enough.
Stream ids should start at roughly 0 and there is no cap.
```Rust
    lines!(n_positions, 0, 2),
```
Primitive macro syntax:
Value or a tuple describing a single point, axis id (not implemented), sub plot id, binds to specific sub plot.
Color is picked based on a hash generated from an expression passed as the first argument.
### Examples
```Rust
// portfolio_backtester_2.rs
CTX.debug(0, 0, timestamp_ns).add((
    lines!(n_positions, 0, 2),
    lines!(n_long, 0, 2),
    lines!(n_short, 0, 2),
    lines!(trades, 0, 2),
    lines!(signal_score, 0, 3),
    lines!(signal_n_samples, 0, 3),
    lines!(signal_risk, 0, 4),
    lines!(avg_lifetime_d, 0, 3),
    texts!((balance, s), 0, 1), // displays a string at y position (balance)
));
// plot_2.rs
CTX.debug(0, 0, timestamp_ns).add((
    candles!((open[i], high[i], low[i], close[i]), 0, 0),
    lines!(long_log_return, 0, 1),
    lines!(short_log_return, 0, 1),
    lines!(long_dd_d, 0, 2),
    lines!(short_dd_d, 0, 2),
    lines!(elapsed_d, 0, 3),
));
// strategy_plot.rs
CTX.debug(0, 1, start_timestamp_ns).add((
    bars!(elapsed_d, 0, 2), // columns ranging from 0 to y
    bars!(long_log_return, 0, 3),
    bars!(long_dd_d, 0, 4),
    bars!(long_entry, 0, 5),
));
// portfolio_backtester.rs (example doesn't work, too old)
// Various properties can be changed, see more at `gpu/esl_gpu/src/debug.rs:344`
// Property is the same for the whole stream
CTX.debug(1, 5, timestamp_ns).add((points!(close_short, 0, 0, |x| x
    .color(Color::RED)
    .shape(PointShape::Cross)
    .radius(5.)),));
```
## UI usage tips
Scroll - zoom
Left click + drag - pan
Right click + drag - zoom to area
Double left click (or press letter r) - reset chart
Series can be hidden by clicking on a legend.
Known bugs:
By default, plot is in streaming mode. In streaming mode, char automatically zooms out to fit all data. While there are small sample sizes, there might be jitter or series will disappear over couple of seconds. Perform any action to exit streaming mode. Sometimes pressing and holding r might stabilize series disappearing after streaming is done.
Candle and bar width isn't calculated correctly when view is at edges of data, zoomed in and mostly displaying nothing.
The whole series might disappear if it contains non finite floating point numbers.
