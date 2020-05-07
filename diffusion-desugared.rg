import "regent"

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = terralib.includecstring[[
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
]]

local SIM = terralib.includec('diffusion.h')
terralib.linklibrary('libdiffusion.so')
-------------------------------------------------------------------------------
-- STOCHASTIC VARIABLE
-------------------------------------------------------------------------------
local K = 0.17
local NUM_UNCERTAINTIES = 1

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------
local DOMAIN_LENGTH = 1.0 -- length of the domain (starts at 0)
local U_0 = 0.0 -- set left boundary value for unknowns
local U_1 = 0.0 -- set right boundary value for unknowns
local F = -1.0 -- set forcing term value

-------------------------------------------------------------------------------
-- MAIN
-------------------------------------------------------------------------------
-- Solve the 1D diffusion eqn. with an uncertain variable coefficient
-- using finite differences and TDMA.
-- del(k del(u))=f on [0,1] subject to u(0)=0 and u(1)=0
-- Here we set f=-1 and k is a random diffusivity
-------------------------------------------------------------------------------
task main()
  C.printf('\n')
  var args = regentlib.c.legion_runtime_get_input_args()

  var num_grid_points = 0
  var num_GPUs = 0
  for i = 1, args.argc do
    if C.strcmp(args.argv[i], '-num_grid_points') == 0 and i < args.argc-1 then
      num_grid_points = C.atoi(args.argv[i+1])
    end
  end


  q = SIM.diffusion_1d(num_grid_points, NUM_UNCERTAINTIES, K)


  __fence(__execution, __block)
  C.printf('q = %f \n', q)
end

regentlib.saveobj(main, "diffusion.o")
