These scripts are used to read solps vairable data from specific files and then generate a new npy file containing the majority of SOLPS variables needed for post pocessing.
The run_b2uf, go_b2uf, run_b2uf_gfile, and go_b2uf_gfile must be placed in a directory where they can be called by the user anywhere in the cluster. The run_* commands should be used after the end of a SOLPS run to generate the variable file.
If using the gfile versions, a gfile must be placed in the solps run directory before running the run_b2uf_gfile.scr command. 
Additionally, the go and run scripts must be changed to your user name and settings before running the commands. 
