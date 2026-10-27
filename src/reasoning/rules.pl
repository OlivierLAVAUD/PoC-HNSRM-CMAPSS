%% src/reasoning/rules.pl - HPC degradation rules for C-MAPSS FD001
%%
%% Thresholds are derived from training-set percentiles (no test leakage):
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

% 95th percentile of the FD001 training split.
sensor_high_threshold(s2, 643.580).
sensor_high_threshold(s3, 1601.470).
sensor_high_threshold(s4, 1425.670).
sensor_high_threshold(s8, 2388.220).
sensor_high_threshold(s9, 9109.980).
sensor_high_threshold(s11, 48.045).
sensor_high_threshold(s13, 2388.230).
sensor_high_threshold(s14, 8181.405).
sensor_high_threshold(s15, 8.511).
sensor_high_threshold(s17, 396.000).

% 5th percentile of the FD001 training split.
sensor_low_threshold(s7, 551.740).
sensor_low_threshold(s12, 520.040).
sensor_low_threshold(s20, 38.490).
sensor_low_threshold(s21, 23.093).

% 95th percentile of absolute cycle-to-cycle change (rule extraction only).
gradient_threshold(s2, 0.830).
gradient_threshold(s3, 11.030).
gradient_threshold(s4, 11.190).
gradient_threshold(s7, 1.130).
gradient_threshold(s8, 0.080).
gradient_threshold(s9, 11.605).
gradient_threshold(s11, 0.280).
gradient_threshold(s12, 0.830).
gradient_threshold(s13, 0.080).
gradient_threshold(s14, 8.650).
gradient_threshold(s15, 0.056).
gradient_threshold(s17, 3.000).
gradient_threshold(s20, 0.280).
gradient_threshold(s21, 0.167).

% ---------------------------------------------------------------------------
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
    Value > 1414.555.

%% s4 (T50) above the 95th percentile.
hpc_temp_critical :-
    current_sensor_value(s4, Value),
    Value > 1425.670.

%% s11 (Ps30, HPC outlet static pressure) above the 75th percentile.
hpc_static_pressure_high :-
    current_sensor_value(s11, Value),
    Value > 47.700.

%% s7 (P30, HPC outlet pressure) below the 25th percentile.
hpc_outlet_pressure_low :-
    current_sensor_value(s7, Value),
    Value < 552.810.

%% s12 (phi, fuel flow ratio) below the 25th percentile.
fuel_flow_low :-
    current_sensor_value(s12, Value),
    Value < 520.960.

%% s21 (W32, LPT coolant bleed) below the 25th percentile.
coolant_bleed_low :-
    current_sensor_value(s21, Value),
    Value < 23.222.

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
