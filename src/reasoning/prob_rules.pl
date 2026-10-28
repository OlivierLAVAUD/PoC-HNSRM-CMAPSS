% src/reasoning/prob_rules.pl - ProbLog model for FD001 HPC failure risk.
% Boolean conditions (hpc_temp_elevated, hpc_static_pressure_high, ...) are
% supplied by prob_query.py from sensor evidence. The default clauses below
% make every condition false unless explicit evidence is appended.

% Default evidence: inactive unless prob_query.py adds the corresponding fact.
hpc_temp_elevated :- fail.
hpc_temp_critical :- fail.
hpc_static_pressure_high :- fail.
hpc_outlet_pressure_low :- fail.
fuel_flow_low :- fail.
coolant_bleed_low :- fail.

% Probability that a deterministic indicator translates into a degraded ratio.
0.80::high_temp_ratio :- hpc_temp_elevated.
0.85::high_pressure_ratio :- hpc_static_pressure_high.
0.90::critical_temp_ratio :- hpc_temp_critical.
0.70::low_outlet_pressure_ratio :- hpc_outlet_pressure_low.
0.65::low_fuel_flow_ratio :- fuel_flow_low.
0.60::low_coolant_ratio :- coolant_bleed_low.

% Combined degradation pathway.
0.55::hpc_degradation :-
    high_temp_ratio,
    high_pressure_ratio,
    low_outlet_pressure_ratio.

% Failure pathways (independent causes, combined by ProbLog).
0.75::failure_risk :- hpc_degradation.
0.85::failure_risk :- critical_temp_ratio, high_pressure_ratio, low_coolant_ratio.
0.60::failure_risk :- high_temp_ratio, low_fuel_flow_ratio.
% Pressure/flow pathway: high static pressure, low outlet pressure and low
% fuel flow also indicate HPC degradation when the outlet temperature is not
% elevated, so that state must not report a zero failure risk.
0.60::failure_risk :-
    high_pressure_ratio,
    low_outlet_pressure_ratio,
    low_fuel_flow_ratio.

query(failure_risk).
query(hpc_degradation).
