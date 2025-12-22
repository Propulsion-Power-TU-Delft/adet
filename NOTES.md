# Future wishlist
- Make instantiating equations of state within an equation more flexible
- Rework argument injection for unit validation, it is messy and needs too many exceptions
    - Assume functions use casadi objects as inputs
    - OR add unit support to equations of state
- Add units inference using regexp 
    - e.g. delta_tot_hmass.* => Always assigned to J / kg
