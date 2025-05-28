import tools
tools.success("Module loading...")
import argparse
import pathlib
import signal
import shlex
import sys
import torch
import experiments
import numpy as np

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")
print("print:Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandline():
  """ Parse the command-line and perform checks.
  Returns:
    Parsed configuration
  """
  # Description
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("--result-directory",
    type=str,
    default="results-data/acp",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot/acp",
    help="Path of the plot directory, containing the graphs traced from the experiments")
  parser.add_argument("--devices",
    type=str,
    default="auto",
    help="Comma-separated list of devices on which to run the experiments, used in a round-robin fashion")
  parser.add_argument("--supercharge",
    type=int,
    default=1,
    help="How many experiments are run in parallel per device, must be positive")
  # Parse command line
  return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
  args = process_commandline()
  # Check the "supercharge" parameter
  if args.supercharge < 1:
    tools.fatal(f"Expected a positive supercharge value, got {args.supercharge}")
  # Make the result directories
  def check_make_dir(path):
    path = pathlib.Path(path)
    if path.exists():
      if not path.is_dir():
        tools.fatal(f"Given path {str(path)!r} must point to a directory")
    else:
      path.mkdir(mode=0o755, parents=True)
    return path
  args.result_directory = check_make_dir(args.result_directory)
  args.plot_directory = check_make_dir(args.plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))


# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "peerToPeerSparseAVG.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)


plot_results = True # To prevent to many plots in the results-plot directory
# Base parameters for the Averaging experiments
params_averaging = {
  "evaluation-delta": 1,
  "nb-steps": 100,
  "nb-honests":26,
  "nb-params":5,
  "random-init":"normal_2", # Here 'normal' intialize nodes'parameter with N(0,I_5), #'normal_2' add +- (5,0,0,0,0) depending on the index
  "topology-name" : "two_worlds",
  "topology-weights": "metropolis",
  "topology-hyper":8,
  }

# Hyperparameters to test
topology_hypers = [params_averaging["topology-hyper"]]
gars = ["CSplus_RG", "GTS_RG", "CShe_RG", 'IOS']
attacks = ["sparse_little", "dissensus", "spectral" , "sparse_empire"]
dataset = "averaging"
params_common = params_averaging
byzcounts = [i for i in range(1,16)]
seeds = tuple(range(6))

result_directory = str(args.result_directory)
plot_directory = str(args.plot_directory)
params = params_common.copy()
params["nb-workers"] = params["nb-honests"]

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge, seeds=seeds)
seeds = jobs.get_seeds()
## Run Jobs
## .... wait for it
for top_hyper in topology_hypers:
  for f in byzcounts:
    for attack in attacks:
      for gar in gars:
          params = params_common.copy()
          params["nb-workers"]=params["nb-honests"] + f
          params["nb-decl-byz"] = f
          params["nb-real-byz"] = f
          params["gar"] = gar
          params["attack"] = attack
          params["topology-hyper"] = top_hyper
          name_experiment = f"{dataset}-{attack}-{gar}-f_{f}-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}"
          jobs.submit(
              name_experiment,
              make_command(params))
  
# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# Import additional modules
try:
 import numpy
 import pandas
 import study
except ImportError as err:
 tools.fatal(f"Unable to plot results: {err}")


def compute_avg_err_op(name, location, *colops, avgs="", errs="-err"):
  """ Compute the average and standard deviation of the selected columns over the given experiment.
  Args:
    name Given experiment name
    location Script to read from
    ...  Tuples of (selected column name (through 'study.select'), optional reduction operation name)
    avgs Suffix for average column names
    errs Suffix for standard deviation (or "error") column names
  Returns:
    Data frames for each of the computed columns,
    Tuple of reduced values per seed (or None if None was provided for 'op')
  Raises:
    'RuntimeError' if a reduction operation was specified for a column selector that did not select exactly 1 column
  """
# Load all the runs for the given experiment name, and keep only a subset
  datas = tuple(study.select(study.Session(result_directory + "/" + name + "-" +str(seed), location), *(col for col, _ in colops)) for seed in seeds)

  # Make the aggregated data frames
  def make_df_ro(col, op):
    nonlocal datas
    # For every selected columns
    subds = tuple(study.select(data, col) for data in datas)
    df    = pandas.DataFrame(index=subds[0].index)
    ro    = None
    for cn in subds[0]:
      # Generate compound column names
      avgn = cn + avgs
      errn = cn + errs
      # Compute compound columns
      numds = numpy.stack(tuple(subd[cn].to_numpy() for subd in subds))
      df[avgn] = numds.mean(axis=0)
      df[errn] = numds.std(axis=0)
      # Compute reduction, if requested
      if op is not None:
        if ro is not None:
          raise RuntimeError(f"column selector {col!r} selected more than one column ({(', ').join(subds[0].columns)}) while a reduction operation was requested")
        ro = tuple(getattr(subd[cn], op)().item() for subd in subds)
    # Return the built data frame and optional computed reduction
    return df, ro
  dfs = list()
  ros = list()
  for col, op in colops:
    df, ro = make_df_ro(col, op)
    #df = df.replace(np.nan, np.inf)
    #df.dropna()
    dfs.append(df)
    ros.append(ro)
  # Return the built data frames and optional computed reductions
  return dfs, ros

algorithms_name = {
          "CSplus_RG":r"CS$^+$-RG",
          "GTS_RG":r"GTS-RG",
          "IOS":r"IOS",
          "CShe_RG":r"ClippedGossip",
          "dsgd":r"D-SGD",
          "cva":r"GTS-RG",
          "trmean":"BRIDGE",
          "rfa":"MoGM",
          "centeredclip":r"ClippedGossip",
          "cgplus":r"CS$^+$-RG"
        }

tools.success("Ploting the accuracy...")
# Plot results
with tools.Context("mnist", "info"):
  if plot_results:
    # Attacks
    for f in byzcounts:
      for top_hyper in topology_hypers:
        for attack in attacks:
            attacked = dict()
            plot = study.LinePlot()
            #plot.include(dsgd[0], "Accuracy", errs="-err", lalp=0.8)
            legend = []
                
            for gar in gars:
                #Gar generic
                name_experiment = f"{dataset}-{attack}-{gar}-f_{f}-h_{params['nb-honests']}-{params["topology-name"]}_{top_hyper}"
                
                cols, _ = compute_avg_err_op(name_experiment, "eval", ("Average loss", "max"))
                mask = cols[0].isnull().any(axis=1)
                cols[0].loc[mask] = [100,0]
                attacked[(gar)] = cols

                plot.include(attacked[(gar)][0], "Average loss", errs="-err", lalp=0.8)
                plot._ax.set_yscale('log')
                legend.append(algorithms_name[gar])

        # plot every time graph in terms of the maximum number of steps
            plot.finalize(None, "Iteration", "MSE", xmin=0, xmax=params_common['nb-steps'], legend=legend)
            plot.save(plot_directory + "/"  + "acp" +"_" + params['topology-name'] + "_" + str(top_hyper) + "_" + attack + "_f=" + str(f) + "-mse" + ".pdf", xsize=3, ysize=1.5)
