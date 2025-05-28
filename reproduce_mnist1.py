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
    default="results-data",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot",
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
# Serial preloading of the dataset
tools.success("Pre-downloading datasets...")

# Pre-load the datasets to prevent the first parallel runs from downloading them several times
with tools.Context("dataset", "info"):
  for name in ("mnist", "cifar10"):
    with tools.Context(name, "info"):
      experiments.make_datasets(name)

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "peerToPeerSparse.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs

result_directory = str(args.result_directory)
plot_directory = str(args.plot_directory)

# Base parameters for the MNIST experiments
params_mnist = {
  "batch-size": 64,
  "model": "simples-minimalcnn",
  "loss": "nll",
  # "l2-regularize": 1e-4,
  "evaluation-delta": 10,
  "subsampled-evaluation":2,
  "nb-steps": 200,
  "momentum-at": "worker",
  "momentum":0.9,
  "dampening":0.9,
  "learning-rate": 0.1,
  "learning-rate-decay-delta": 40,
  "learning-rate-decay": 40,
  "dataset": "mnist",
  "numb-labels": 10,
  "nb-honests":20,
  "topology-name" : "Erdos_Renyi",
  "topology-weights": "metropolis",
  "dirichlet-alpha":1
  }
  

# Hyperparameters to test
gars = ["CSplus_RG", "GTS_RG", "CShe_RG", 'IOS']
attacks = ["dissensus", "spectral" ,"sparse_little", "sparse_empire"]#, "signflipping", "labelflipping"]
dataset = params_mnist["dataset"]
byzcounts = [4]
topology_hypers = list(np.linspace(0.26,1,11))


alpha = params_mnist["dirichlet-alpha"]
momentum = params_mnist['momentum']
params_mnist["dampening"] = params_mnist["momentum"]

params_common = params_mnist

# Jobs

seeds=tuple(range(1))
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge, seeds=seeds)
seeds = jobs.get_seeds()


# Submit all experiments
for top_hyper in topology_hypers:
    params = params_common.copy()
    params['topology-hyper'] = top_hyper
    params["nb-workers"] = params["nb-honests"]
    params['gar'] = 'average_sparse'
    # DSGD
    name_experiment = f"{dataset}-average_sparse-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}-m_{momentum}-alpha_{alpha}_dsgd"
    jobs.submit(name_experiment,make_command(params))

    # Attacks
    for f in byzcounts:
        for attack in attacks:
            for gar in gars:
                    params = params_common.copy()
                    params['topology-hyper'] = top_hyper
                    params["nb-workers"]=params["nb-honests"] + f
                    params["nb-decl-byz"] = f
                    params["nb-real-byz"] = f
                    params["gar"] = gar
                    params["attack"] = attack
                    name_experiment = f"{dataset}-{attack}-{gar}-f_{f}-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}-m_{momentum}-alpha_{alpha}"
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
  for top_hyper in topology_hypers:
    params = params_common.copy()
    params['topology-hyper'] = top_hyper

    # DSGD
    name_experiment = f"{dataset}-average_sparse-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}-m_{momentum}-alpha_{alpha}_dsgd"
    plot_dsgd = True
    try:
      dsgd, _ = compute_avg_err_op(name_experiment, "eval", ("Average loss", "max"))
    except Exception as err:
      tools.warning(f"Unable to process {name_experiment}: {err}")
      plot_dsgd = False
  

    for f in byzcounts:
        for attack in attacks:
            attacked = dict()
            plot = study.LinePlot()
            if plot_dsgd:
              plot.include(dsgd[0], "Accuracy", errs="-err", lalp=0.8)
              legend = ["D-SGD"]
            else:
               legend = []
            
            for gar in gars:
                #Gar generic
                name_experiment = f"{dataset}-{attack}-{gar}-f_{f}-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}-m_{momentum}-alpha_{alpha}"
                try:
                    cols, _ = compute_avg_err_op(name_experiment, "eval", ("Accuracy", "max"))
                    attacked[(gar, momentum)] = cols
                    
                except Exception as err:
                    tools.warning(f"Unable to process {name_experiment !r}: {err}")
                    continue

                # Plot top-1 cross-accuracies

                plot.include(attacked[(gar, momentum)][0], "Accuracy", errs="-err", lalp=0.8)
                legend.append(algorithms_name[gar])

            # plot every time graph in terms of the maximum number of steps
            plot.finalize(None, "Iteration", "Test accuracy", xmin=0, xmax=params_common['nb-steps'], ymin=0, ymax=1, legend=legend)
            plot.save(plot_directory + "/"  + dataset +"_" + params['topology-name'] + "_" + params_common['model'] + "_" + attack + "_f=" + str(f) + "_alpha=" + str(alpha)+ "-acc" + ".pdf", xsize=3, ysize=1.5)

    tools.success("Ploting the train loss...")
    ## ploting the loss

    # DSGD
    name_experiment = f"{dataset}-average_sparse-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}-m_{momentum}-alpha_{alpha}_dsgd"
    plot_dsgd = True
    try:
      dsgd, _ = compute_avg_err_op(name_experiment, "eval", ("Average loss", "max"))
    except Exception as err:
      tools.warning(f"Unable to process {name_experiment}: {err}")
      plot_dsgd = False

    for f in byzcounts:
        for attack in attacks:
            attacked = dict()
            plot = study.LinePlot()
            if plot_dsgd:
              plot.include(dsgd[0], "Average loss", errs="-err", lalp=0.8)
              legend = ["D-SGD"]
            else:
               legend = []
            
            for gar in gars:
                #Gar generic
                name_experiment = f"{dataset}-{attack}-{gar}-f_{f}-h_{params['nb-honests']}-{params["topology-name"]}_{params["topology-hyper"]}-m_{momentum}-alpha_{alpha}"
                try:
                    cols, _ = compute_avg_err_op(name_experiment, "eval", ("Average loss", "max"))
  
                    mask = cols[0].isnull().any(axis=1)
                    cols[0].loc[mask] = [100,0]
                    attacked[(gar, momentum)] = cols
                except Exception as err:
                    tools.warning(f"Unable to process {name_experiment !r}: {err}")
                    continue

                # Plot top-1 cross-accuracies

                plot.include(attacked[(gar, momentum)][0], "Average loss", errs="-err", lalp=0.8)
                legend.append(algorithms_name[gar])

            # plot every time graph in terms of the maximum number of steps"
            plot.finalize(None, "Iteration", "Train loss", xmin=0, xmax=params_common['nb-steps'], ymin=0, ymax=3, legend=legend)
            plot.save(plot_directory + "/" + dataset +"_" + params['topology-name']+ "_" + params_common['model'] + "_" + attack + "_f=" + str(f) + "_alpha=" + str(alpha) + "-train_loss" + ".pdf", xsize=3, ysize=1.5)
