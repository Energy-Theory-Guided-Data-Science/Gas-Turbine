# Data_Understanding
This file should help us understand all the data provided by the gas turbine expert

## Raw Data
There are multiple files provided. The ones relevant for us are "Leistungdaten_" and "Drehzahldaten_" which contain the relevant data for the output powers and the spinning number respectively. Both files are already prepared by the expert. Maybe it is relevant to look at the raw data at some point but until then the prepared data seems good enough.

Further relevant are the data starting with 'Daten_Test_ID_'. They contain the measurements regarding thermical and electrical output as well as the regarding time frames. In these files the experiment setting was 0-30-50-30-75-30-100-0.
The .mat files named n_soll, P_el_rms, P_th and t_nsoll, t_el_rms, t_th respectively are the spining number (Drehzahl), electrical and thermal output with their time frames respectively for a different experiment where the input followed the 0-100-30-100-30-0 structure.

Files having the ".m" file extension are MatLab Code Files, which are implemented by the expert and can be used to further understand the relation between different columns.


## Relation between them
The thermal power is calculated by
	P_th  =  V_dotKW  \*  rho_KW  \*  c_KW55  \*  (T_vorKW  -  T_rueckKW)


The electrical power is calculated by
	P_el_rms = T_N15_rms \* U_N_rms