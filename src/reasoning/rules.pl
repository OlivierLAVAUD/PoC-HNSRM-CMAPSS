%% src/reasoning/rules.pl - HPC degradation rules for C-MAPSS FD001
%%
%% Thresholds are derived from the 20 training engines selected by the
%% pipeline's 20/20/60 unit split (random_state=42), before any validation or
%% held-out engine is seen:
%%   - "elevated" / "high"  : 75th percentile for early warning
%%   - "critical"           : 95th percentile for severe degradation
%%   - "low"                : 25th percentile for reversed indicators
%% Sensors rising with degradation: s2, s3, s4, s8, s9, s11, s13, s14, s15, s17.
%% Sensors falling with degradation: s7, s12, s20, s21.

:- discontiguous(neural_rule/0).
:- discontiguous(gradient_rule/0).
:- discontiguous(pattern_rule/0).
:- discontiguous(abstract_pattern/0).

:- dynamic current_sensor_value/2.
:- dynamic sensor_change/2.
:- dynamic sensor_delta/2.
:- dynamic sensor_slope/2.
:- dynamic sensor_relative_change/2.
:- dynamic sensor_moving_average/2.
:- dynamic sensor_std/2.
:- dynamic sensor_persistence/3.
:- dynamic temporal_threshold/3.

% ---------------------------------------------------------------------------
% Sensor direction and training-derived thresholds
% ---------------------------------------------------------------------------
sensor_direction(s2, higher_worse).
sensor_direction(s3, higher_worse).
sensor_direction(s4, higher_worse).
sensor_direction(s7, lower_worse).
sensor_direction(s8, higher_worse).
sensor_direction(s9, higher_worse).
sensor_direction(s11, higher_worse).
sensor_direction(s12, lower_worse).
sensor_direction(s13, higher_worse).
sensor_direction(s14, higher_worse).
sensor_direction(s15, higher_worse).
sensor_direction(s17, higher_worse).
sensor_direction(s20, lower_worse).
sensor_direction(s21, lower_worse).

% 95th percentile of the 20 FD001 training engines.
sensor_high_threshold(s2, 643.590).
sensor_high_threshold(s3, 1601.330).
sensor_high_threshold(s4, 1426.015).
sensor_high_threshold(s8, 2388.220).
sensor_high_threshold(s9, 9105.077).
sensor_high_threshold(s11, 48.050).
sensor_high_threshold(s13, 2388.230).
sensor_high_threshold(s14, 8178.803).
sensor_high_threshold(s15, 8.510).
sensor_high_threshold(s17, 396.000).

% 5th percentile of the 20 FD001 training engines.
sensor_low_threshold(s7, 551.710).
sensor_low_threshold(s12, 520.021).
sensor_low_threshold(s20, 38.490).
sensor_low_threshold(s21, 23.093).

% Named HPC rule thresholds. This block is the single source of truth for the
% six degradation rules below; src/reasoning/prob_query.py reads these facts.
% elevated / high = 75th percentile, critical = 95th percentile, low = 25th,
% measured on the 20 training engines.
hpc_rule_threshold(s4, elevated, 1414.993).
hpc_rule_threshold(s4, critical, 1426.015).
hpc_rule_threshold(s11, high, 47.710).
hpc_rule_threshold(s7, low, 552.777).
hpc_rule_threshold(s12, low, 520.930).
hpc_rule_threshold(s21, low, 23.218).

% 95th percentile of absolute cycle-to-cycle change on the 20 training
% engines (rule extraction only).
gradient_threshold(s2, 0.810).
gradient_threshold(s3, 11.018).
gradient_threshold(s4, 11.047).
gradient_threshold(s7, 1.138).
gradient_threshold(s8, 0.080).
gradient_threshold(s9, 11.518).
gradient_threshold(s11, 0.280).
gradient_threshold(s12, 0.840).
gradient_threshold(s13, 0.080).
gradient_threshold(s14, 8.558).
gradient_threshold(s15, 0.056).
gradient_threshold(s17, 3.000).
gradient_threshold(s20, 0.280).
gradient_threshold(s21, 0.166).

% ---------------------------------------------------------------------------
% Feature name aliases: map FD001 sensor numbers to human-readable feature names
% These allow neural rule extraction to use descriptive names (temperature,
% vibration, pressure, efficiency_index) instead of raw sensor numbers (s4, s11, s7, s12).
% ---------------------------------------------------------------------------
temperature_alias(s4).   % T50, LPT outlet temperature
vibration_alias(s11).    % Ps30, HPC static pressure (proxy)
pressure_alias(s7).      % P30, HPC outlet pressure (proxy)
efficiency_alias(s12).   % phi, fuel flow (proxy)
% Generic predicates used by neural rule extraction
% ---------------------------------------------------------------------------
feature_threshold(Sensor, Value, high) :-
    sensor_direction(Sensor, higher_worse),
    current_sensor_value(Sensor, Value),
    sensor_high_threshold(Sensor, Threshold),
    Value > Threshold.

feature_threshold(Sensor, Value, low) :-
    sensor_direction(Sensor, lower_worse),
    current_sensor_value(Sensor, Value),
    sensor_low_threshold(Sensor, Threshold),
    Value < Threshold.

feature_gradient(Sensor, Gradient, high) :-
    sensor_change(Sensor, Gradient),
    gradient_threshold(Sensor, Threshold),
    Gradient > Threshold.

trend(Sensor, increasing) :-
    sensor_delta(Sensor, Delta),
    Delta > 0.

trend(Sensor, decreasing) :-
    sensor_delta(Sensor, Delta),
    Delta < 0.

% ---------------------------------------------------------------------------
% HPC degradation rules (FD001)
% ---------------------------------------------------------------------------
%% s4 (T50, LPT outlet temperature) above the 75th percentile.
hpc_temp_elevated :-
    current_sensor_value(s4, Value),
    hpc_rule_threshold(s4, elevated, Threshold),
    Value > Threshold.

%% s4 (T50) above the 95th percentile.
hpc_temp_critical :-
    current_sensor_value(s4, Value),
    hpc_rule_threshold(s4, critical, Threshold),
    Value > Threshold.

%% s11 (Ps30, HPC outlet static pressure) above the 75th percentile.
hpc_static_pressure_high :-
    current_sensor_value(s11, Value),
    hpc_rule_threshold(s11, high, Threshold),
    Value > Threshold.

%% s7 (P30, HPC outlet pressure) below the 25th percentile.
hpc_outlet_pressure_low :-
    current_sensor_value(s7, Value),
    hpc_rule_threshold(s7, low, Threshold),
    Value < Threshold.

%% s12 (phi, fuel flow ratio) below the 25th percentile.
fuel_flow_low :-
    current_sensor_value(s12, Value),
    hpc_rule_threshold(s12, low, Threshold),
    Value < Threshold.

%% s21 (W32, LPT coolant bleed) below the 25th percentile.
coolant_bleed_low :-
    current_sensor_value(s21, Value),
    hpc_rule_threshold(s21, low, Threshold),
    Value < Threshold.

%% Combined HPC degradation signature.
hpc_degradation_pattern :-
    hpc_temp_elevated,
    hpc_static_pressure_high,
    hpc_outlet_pressure_low.

%% Severe HPC signature.
hpc_critical_pattern :-
    hpc_temp_critical,
    hpc_static_pressure_high,
    coolant_bleed_low.

%% Thermal overload without coolant compensation.
hpc_thermal_overload :-
    hpc_temp_critical,
    hpc_static_pressure_high.

%% Fuel starvation with reduced coolant bleed.
hpc_fuel_starvation :-
    fuel_flow_low,
    coolant_bleed_low.

% ---------------------------------------------------------------------------
% Patterns reused by neural rule extraction
% ---------------------------------------------------------------------------
correlated(s4, s11) :-
    hpc_temp_elevated,
    hpc_static_pressure_high.

sequence_pattern(hpc_degradation_trend) :-
    trend(s4, increasing),
    trend(s11, increasing),
    trend(s7, decreasing).

% ---------------------------------------------------------------------------
% Temporal window features (thresholds are asserted from training windows)
% ---------------------------------------------------------------------------
temporal_trend(Sensor, increasing) :-
    sensor_slope(Sensor, Slope),
    temporal_threshold(Sensor, slope_abs, SlopeThreshold),
    Slope > SlopeThreshold,
    sensor_relative_change(Sensor, Relative),
    temporal_threshold(Sensor, relative_change_abs, RelativeThreshold),
    Relative > RelativeThreshold.

temporal_trend(Sensor, decreasing) :-
    sensor_slope(Sensor, Slope),
    temporal_threshold(Sensor, slope_abs, SlopeThreshold),
    Slope < -SlopeThreshold,
    sensor_relative_change(Sensor, Relative),
    temporal_threshold(Sensor, relative_change_abs, RelativeThreshold),
    Relative < -RelativeThreshold.

temporal_persistence(Sensor, above) :-
    sensor_persistence(Sensor, above, Count),
    temporal_threshold(Sensor, persistence, Minimum),
    Count >= Minimum.

temporal_persistence(Sensor, below) :-
    sensor_persistence(Sensor, below, Count),
    temporal_threshold(Sensor, persistence, Minimum),
    Count >= Minimum.

temporal_high_variability(Sensor) :-
    sensor_std(Sensor, Std),
    temporal_threshold(Sensor, std, StdThreshold),
    Std > StdThreshold.

temporal_moving_average_high(Sensor) :-
    sensor_moving_average(Sensor, Average),
    sensor_high_threshold(Sensor, Threshold),
    Average > Threshold.

temporal_thermal_pressure_degradation :-
    temporal_trend(s4, increasing),
    ( temporal_trend(s7, decreasing)
    ; temporal_persistence(s11, above)
    ).
