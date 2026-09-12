from netCDF4 import Dataset
import numpy as np
import sys

if len(sys.argv) > 1:
    print("Arguments passed:")
    for i, arg in enumerate(sys.argv[1:]):
        print(f"  Argument {i+1}: {arg}")
else:
    print("No arguments passed.")


def convert_netcdf4_to_netcdf3(input_file, output_file):
    """
    Converts a NetCDF4 file to NetCDF3 format.

    Args:
        input_file (str): Path to the input NetCDF4 file.
        output_file (str): Path for the output NetCDF3 file.
    """
    try:
        # Open the NetCDF4 input file in read mode
        with Dataset(input_file, 'r') as nc4_dataset:
            # Create a new NetCDF3 output file in write mode
            # Specify the format as 'NETCDF3_CLASSIC' or 'NETCDF3_64BIT_OFFSET'
            # NETCDF3_CLASSIC has a 2GB file size limit.
            # NETCDF3_64BIT_OFFSET supports larger files but older readers might not support it.
            with Dataset(output_file, 'w', format='NETCDF3_CLASSIC') as nc3_dataset:

                # Copy global attributes
                for attr_name in nc4_dataset.ncattrs():
                    setattr(nc3_dataset, attr_name, getattr(nc4_dataset, attr_name))

                # Copy dimensions
                for dim_name, dimension in nc4_dataset.dimensions.items():
                    nc3_dataset.createDimension(dim_name, len(dimension) if not dimension.isunlimited() else None)

                # Copy variables and their data
                for var_name, variable in nc4_dataset.variables.items():
                    
                    if var_name in ['curvature_type','hypnotoad_inputs','hypnotoad_inputs_yaml','hypnotoad_input_geqdsk_file_contents','Python_version','module_versions']:
                        continue

                    # Create the variable in the NetCDF3 file
                    new_variable = nc3_dataset.createVariable(var_name, variable.dtype, variable.dimensions)

                    # Copy variable attributes
                    for attr_name in variable.ncattrs():
                        setattr(new_variable, attr_name, getattr(variable, attr_name))

                    # Copy variable data
                    new_variable[:] = variable[:]

        print(f"Successfully converted '{input_file}' to NetCDF3 format as '{output_file}'.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# Example usage:
# Create a dummy NetCDF4 file for demonstration

# Convert the dummy NetCDF4 file to NetCDF3
convert_netcdf4_to_netcdf3(str(sys.argv[1]),str(sys.argv[2]))

