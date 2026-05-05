from adet.equations.varspec import VarSpec
from adet.equations.variables import VariableEnum


class Losses(VariableEnum):
    Dht_loading = VarSpec(
        'dht_loading', 'Enthalpy generation (blade loading)', 'J / kg'
    )
    Dht_clearance = VarSpec(
        'dht_clearance', 'Enthalpy generation (tip clearance)', 'J / kg'
    )
    Dht_skin = VarSpec('dht_skin', 'Enthalpy generation (skin friction)', 'J / kg')
    Dht_incidence = VarSpec(
        'dht_incidence', 'Enthalpy generation (incidence)', 'J / kg'
    )
    Dht_mixing = VarSpec('dht_mixing', 'Enthalpy generation (mixing)', 'J / kg')
    Dht_disk = VarSpec('dht_disk', 'Enthalpy generation (disk friction)', 'J / kg')
    Dht_recirculation = VarSpec(
        'dht_recirculation', 'Enthalpy generation (recirculation)', 'J / kg'
    )
    Dht_leakage = VarSpec('dht_leakage', 'Enthalpy generation (leakage)', 'J / kg')
    Dht_lost = VarSpec('dht_lost', 'Enthalpy generation (leakage work loss)', 'J / kg')
    # Entropy based
    Ds_leakage = VarSpec('ds_leakage', 'Entropy generation (leakage)', 'J / kg / K')
    Ds_mixing = VarSpec('ds_mixing', 'Entropy generation (mixing)', 'J / kg / K')
    Ds_profile = VarSpec('ds_profile', 'Entropy generation (profile)', 'J / kg / K')
    Ds_secondary = VarSpec(
        'ds_secondary', 'Entropy generation (secondary)', 'J / kg / K'
    )
    Dtot_hmass = VarSpec('dtot_hmass', 'Total enthalpy loss', 'J / kg')
