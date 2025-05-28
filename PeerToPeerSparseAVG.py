
import tools
tools.success("Module P2P loading...")
import argparse
import collections
import json
import math
import os
import pathlib
import random
import signal
import sys
import torch
import torchvision
import traceback
import aggregators
import attacks
import experiments
import networkx as nx
import topology

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous P2P initializations...")
print("Print:Miscellaneous P2P initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line P2P processing...")

def process_commandline():
	""" Parse the command-line and perform checks.
	Returns:
		Parsed configuration
	"""
	# Description
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--seed",
		type=int,
		default=-1,
		help="Fixed seed to use for reproducibility purpose, negative for random seed")
	parser.add_argument("--device",
		type=str,
		default="auto",
		help="Device on which to run the experiment, \"auto\" by default")
	parser.add_argument("--device-gar",
		type=str,
		default="same",
		help="Device on which to run the GAR, \"same\" for no change of device")
	parser.add_argument("--nb-steps",
		type=int,
		default=1000,
		help="Number of (additional) training steps to do, negative for no limit")
	parser.add_argument("--nb-workers",
		type=int,
		default=15,
		help="Total number of worker machines")
	# if nb-honests is specified (instead of nb-workers)
	parser.add_argument("--nb-honests",
		type=int,
		default=None,
		help="""nb of honest worker, if not specified then defined as nb_worker - nb_byz, 
		if specified then overide nb-worker to ensure that nb-worker = nb_honest + nb_byz"""
		)
	parser.add_argument("--nb-for-study",
		type=int,
		default=1,
		help="Number of gradients to compute for gradient study purpose only, non-positive for no study even when the result directory is set")
	parser.add_argument("--nb-for-study-past",
		type=int,
		default=1,
		help="Number of past gradients to keep for gradient study purpose only, ignored if no study")
	parser.add_argument("--nb-decl-byz",
		type=int,
		default=0,
		help="Number of Byzantine worker(s) to support")
	parser.add_argument("--nb-real-byz",
		type=int,
		default=0,
		help="Number of actual Byzantine worker(s)")
	parser.add_argument("--init-multi",
		type=str,
		default=None,
		help="Model multi-dimensional parameters initialization algorithm; use PyTorch's default if unspecified")
	parser.add_argument("--init-multi-args",
		nargs="*",
		help="Additional initialization algorithm-dependent arguments to pass when initializing multi-dimensional parameters")
	parser.add_argument("--init-mono",
		type=str,
		default=None,
		help="Model mono-dimensional parameters initialization algorithm; use PyTorch's default if unspecified")
	parser.add_argument("--init-mono-args",
		nargs="*",
		help="Additional initialization algorithm-dependent arguments to pass when initializing mono-dimensional parameters")
	parser.add_argument("--gar",
		type=str,
		default="average",
		help="(Byzantine-resilient) aggregation rule to use")
	parser.add_argument("--gar-args",
		nargs="*",
		help="Additional GAR-dependent arguments to pass to the aggregation rule")
	parser.add_argument("--gar-pivot",
		type=str,
		default=None,
		help="Pivot (gar name) to be used in case of CVA aggregation rule")
	parser.add_argument("--gar-second",
		type=str,
		default=None,
		help="Second (Byzantine-resilient) aggregation rule to use on top of bucketing or NNM")
	parser.add_argument("--bucket-size",
		type=int,
		default=1,
		help="Size of buckets (i.e., number of gradients to average per bucket) in case of bucketing technique")
	parser.add_argument("--gars",
		type=str,
		default=None,
		help="JSON-string specifying several GARs to use randomly at each step; overrides '--gar' and '--gar-args' if specified")
	parser.add_argument("--attack",
		type=str,
		default="nan",
		help="Attack to use")
	parser.add_argument("--attack-args",
		nargs="*",
		help="Additional attack-dependent arguments to pass to the attack")
	parser.add_argument("--model",
		type=str,
		default="simples-full",
		help="Model to train")
	parser.add_argument("--model-args",
		nargs="*",
		help="Additional model-dependent arguments to pass to the model")
	parser.add_argument("--loss",
		type=str,
		default="nll",
		help="Loss to use")
	parser.add_argument("--loss-args",
		nargs="*",
		help="Additional loss-dependent arguments to pass to the loss")
	parser.add_argument("--criterion",
		type=str,
		default="top-k",
		help="Criterion to use")
	parser.add_argument("--criterion-args",
		nargs="*",
		help="Additional criterion-dependent arguments to pass to the criterion")
	parser.add_argument("--dataset",
		type=str,
		default="mnist",
		help="Dataset to use")
	parser.add_argument("--dataset-length",
		type=str,
		default=60000,
		help="Length of dataset to use")
	parser.add_argument("--batch-size",
		type=int,
		default=25,
		help="Batch-size to use for training")
	parser.add_argument("--batch-size-test",
		type=int,
		default=100,
		help="Batch-size to use for testing")
	parser.add_argument("--batch-size-test-reps",
		type=int,
		default=100,
		help="How many evaluation(s) with the test batch-size to perform")
	parser.add_argument("--no-transform",
		action="store_true",
		default=False,
		help="Whether to disable any dataset tranformation (e.g. random flips)")
	parser.add_argument("--learning-rate",
		type=float,
		default=0.5,
		help="Learning rate to use for training")
	parser.add_argument("--learning-rate-decay",
		type=int,
		default=5000,
		help="Learning rate hyperbolic half-decay time, non-positive for no decay")
	parser.add_argument("--learning-rate-decay-delta",
		type=int,
		default=1,
		help="How many steps between two learning rate updates, must be a positive integer")
	parser.add_argument("--learning-rate-schedule",
		type=str,
		default=None,
		help="Learning rate schedule, format: <init lr>[,<from step>,<new lr>]*; if set, supersede the other '--learning-rate' options")
	parser.add_argument("--momentum",
		type=float,
		default=0.99,
		help="Momentum to use for training")
	parser.add_argument("--dampening",
		type=float,
		default=0.99,
		help="Dampening to use for training")
	parser.add_argument("--momentum-at",
		type=str,
		default="worker",
		help="Where to apply the momentum & dampening ('update': just after the GAR, 'server': just before the GAR, 'worker': at each worker)")
	parser.add_argument("--weight-decay",
		type=float,
		default=0,
		help="Weight decay to use for training")
	parser.add_argument("--l1-regularize",
		type=float,
		default=None,
		help="Add L1 regularization of the given factor to the loss")
	parser.add_argument("--l2-regularize",
		type=float,
		default=1e-4,
		help="Add L2 regularization of the given factor to the loss")
	parser.add_argument("--gradient-clip",
		type=float,
		default=None,
		help="Maximum L2-norm, above which clipping occurs, for the estimated gradients")
	parser.add_argument("--gradient-clip-centered",
		default="adaptive",
		help="Maximum L2-norm of the distance between the current and previous gradients, will only be used for centered clipping GAR")
	parser.add_argument("--nb-local-steps",
		type=int,
		default=1,
		help="Positive integer, number of local training steps to perform to make a gradient (1 = standard SGD)")
	parser.add_argument("--load-checkpoint",
		type=str,
		default=None,
		help="Load a given checkpoint to continue the stored experiment")
	parser.add_argument("--result-directory",
		type=str,
		default=None,
		help="Path of the directory in which to save the experiment results (loss, cross-accuracy, ...) and checkpoints, empty for no saving")
	parser.add_argument("--evaluation-delta",
		type=int,
		default=1,
		help="How many training steps between model evaluations, 0 for no evaluation")
	parser.add_argument("--subsampled-evaluation",
		type=int,
		default=1,
		help="On how many nodes the train loss and accuracy are computed")
	parser.add_argument("--checkpoint-delta",
		type=int,
		default=0,
		help="How many training steps between experiment checkpointing, 0 or leave '--result-directory' empty for no checkpointing")
	parser.add_argument("--user-input-delta",
		type=int,
		default=0,
		help="How many training steps between two prompts for user command inputs, 0 for no user input")
	parser.add_argument("--privacy",
		action="store_true",
		default=False,
		help="Gaussian privacy noise ε constant")
	parser.add_argument("--privacy-epsilon",
		type=float,
		default=0.1,
		help="Gaussian privacy noise ε constant; ignore if '--privacy' is not specified")
	parser.add_argument("--privacy-delta",
		type=float,
		default=1e-5,
		help="Gaussian privacy noise δ constant; ignore if '--privacy' is not specified")
	parser.add_argument("--batch-increase",
		action="store_true",
		default=False,
		help="Activate exponential batch increase (experimental functionality)")
	#JS: 2 arguments for coordinate descent
	parser.add_argument("--coordinates",
		type=int,
		default=0,
		help="Number of coordinates for the coordinate SGD. If it is set to 0, then we execute the regular SGD algorithm (with full coordinates)")
    #JS: argument for enbaling MVR
	parser.add_argument("--mvr",
		action="store_true",
		default=False,
		help="Execute the MVR technique on the momentum")
    #JS: argument for enbaling heterogeneity
	parser.add_argument("--hetero",
		action="store_true",
		default=False,
		help="Handle heterogeneous datasets (i.e., one data iterator per worker)")
    #JS: argument for distinct datasets for honest workers
	parser.add_argument("--distinct-data",
		action="store_true",
		default=False,
		help="Distinct datasets for honest workers (e.g., privacy setting)")
    #JS: argument for sampling honest data using Dirichlet distribution
	parser.add_argument("--dirichlet-alpha",
		type=float,
		default=None,
		help="The alpha parameter for distribution the data among honest workers using Dirichlet ")
    #JS: argument for number of labels of heterogeneous dataset
	parser.add_argument("--mimic-heuristic",
		action="store_true",
		default=False,
		help="Use heuristic to determine the best worker to mimic")
    #JS: argument for the heuristic of the mimic attack (enabling the computation of mu and z)
	parser.add_argument("--mimic-learning-phase",
		type=int,
		default=100,
		help="Number of steps in the learning phase of the mimic heuristic attack")
	#: argument for the communication network
	parser.add_argument("--topology-name",
		type=str,
		default="fully_connected",
		help="Name of the graph decribing the communication network")
	parser.add_argument("--topology-weights",
		type=str,
		default="metropolis",
		help="Method used to define edges' weights on the graph")
	parser.add_argument("--topology-hyper",
		type=float,
		default=1,
		help="Hyperparamer 1 of the considered topology")
	parser.add_argument("--random-init",
		type=str,
		default="normal",
		help="Specify the init distribution for the nodes."
		)
	parser.add_argument("--nb-params",
		type=int,
		default=5,
		help="Dimention of the parameter")
    # Parse command line
	return parser.parse_args(sys.argv[1:])


with tools.Context("cmdline", "info"):
	args = process_commandline()
	# Parse additional arguments
	for name in ("init_multi", "init_mono", "gar", "attack", "model", "loss", "criterion"):
		name = f"{name}_args"
		keyval = getattr(args, name)
		setattr(args, name, dict() if keyval is None else tools.parse_keyval(keyval))
	# Count the number of real honest workers
	if args.nb_honests is not None and args.nb_honests > 0:
		args.nb_workers = args.nb_honests + args.nb_real_byz
	else:
		args.nb_honests = args.nb_workers - args.nb_real_byz
		if args.nb_honests < 0:
			tools.fatal(f"Invalid arguments: there are more real Byzantine workers \
			   ({args.nb_real_byz}) than total workers ({args.nb_workers})")
	# Check the learning rate and associated options
	# Check no checkpoint to load if reproducibility requested
	if args.seed >= 0 and args.load_checkpoint is not None:
		tools.warning("Unable to enforce reproducibility when a checkpoint is loaded; ignoring seed")
		args.seed = -1
	# Check at least one gradient in past for studying purpose, or none if study disabled

	if args.result_directory is None:
		if args.nb_for_study > 0:
			args.nb_for_study = 0
		if args.nb_for_study_past > 0:
			args.nb_for_study_past = 0
	else:
		if args.nb_for_study_past < 1:
			tools.warning("At least one gradient must exist in the past to enable studying honest curvature; set '--nb-for-study-past 1'")
			args.nb_for_study_past = 1
		elif math.isclose(args.momentum, 0.0) and args.nb_for_study_past > 1:
			tools.warning("Momentum is (almost) zero, no need to store more than the previous honest gradient; set '--nb-for-study-past 1'")
			args.nb_for_study_past = 1
	# Print configuration
	def cmd_make_tree(subtree, level=0):
		if isinstance(subtree, tuple) and len(subtree) > 0 and isinstance(subtree[0], tuple) and len(subtree[0]) == 2:
			label_len = max(len(label) for label, _ in subtree)
			iterator  = subtree
		elif isinstance(subtree, dict):
			if len(subtree) == 0:
				return " - <none>"
			label_len = max(len(label) for label in subtree.keys())
			iterator  = subtree.items()
		else:
			return f" - {subtree}"
		level_spc = "  " * level
		res = ""
		for label, node in iterator:
			res += f"{os.linesep}{level_spc}· {label}{' ' * (label_len - len(label))}{cmd_make_tree(node, level + 1)}"
		return res
	if args.gars is None:
		cmdline_gars = (
			("Name", args.gar),
			("Arguments", args.gar_args))
	else:
		cmdline_gars = list()
		for info in args.gars.split(";"):
			info = info.split(",", maxsplit=2)
			if len(info) < 2:
				info.append("1")
			if len(info) < 3:
				info.append(None)
			else:
				try:
					info[2] = json.loads(info[2].strip())
				except json.decoder.JSONDecodeError:
					info[2] = "<parsing failed>"
			cmdline_gars.append((f"Frequency {info[1].strip()}", (
				("Name", info[0].strip()),
				("Arguments", info[2]))))
		cmdline_gars = tuple(cmdline_gars)
	cmdline_config = "Configuration" + cmd_make_tree((
		("Reproducibility", "not enforced" if args.seed < 0 else (f"enforced (seed {args.seed})")),
		("#workers", args.nb_workers),
		("#honests", args.nb_honests),
		("topology", args.topology_name),
		("topology weights", args.topology_weights),
		("topology hyper", args.topology_hyper),
		("#local steps", "1 (standard)" if args.nb_local_steps == 1 else f"{args.nb_local_steps}"),
		("#declared Byz.", args.nb_decl_byz),
		("#actually Byz.", args.nb_real_byz),
		("Loss", (
			("Name", "MSE"))),
		("Optimizer", (
			("Name", "averaging"))),
		("Attack", (
			("Name", args.attack),
			("Arguments", args.attack_args))),
		("Aggregation" if args.gars is None else "Aggregations", cmdline_gars),
		("Second Aggregation", args.gar_second if args.gar == "bucketing" else "No bucketing"),
			))
	print(cmdline_config)

# ---------------------------------------------------------------------------- #
# Setup
tools.success("Experiment setup...")

def result_make(name, *fields):
	""" Make and bind a new result file with a name, initialize with a header line.
	Args:
		name    Name of the result file
		fields... Name of each field, in order
	Raises:
		'KeyError' if name is already bound
		'RuntimeError' if no name can be bound
		Any exception that 'io.FileIO' can raise while opening/writing/flushing
	"""
	# Check if results are to be output
	global args
	if args.result_directory is None:
		raise RuntimeError("No result is to be output")
	# Check if name is already bounds
	global result_fds
	if name in result_fds:
		raise KeyError(f"Name {name!r} is already bound to a result file")
	# Make the new file
	fd = (args.result_directory / name).open("w")
	fd.write("# " + ("\t").join(str(field) for field in fields))
	fd.flush()
	result_fds[name] = fd

def result_get(name):
	""" Get a valid descriptor to the bound result file, or 'None' if the given name is not bound.
	Args:
		name Given name
	Returns:
		Valid file descriptor, or 'None'
	"""
	# Check if results are to be output
	global args
	if args.result_directory is None:
		return None
	# Return the bound descriptor, if any
	global result_fds
	return result_fds.get(name, None)

def result_store(fd, *entries):
	""" Store a line in a valid result file.
	Args:
		fd     Descriptor of the valid result file
		entries... Object(s) to convert to string and write in order in a new line
	"""
	fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
	fd.flush()


with tools.Context("setup", "info"):
	# Enforce reproducibility if asked (see https://pytorch.org/docs/stable/notes/randomness.html)
	reproducible = args.seed >= 0
	if reproducible:
		torch.manual_seed(args.seed)
		import numpy
		numpy.random.seed(args.seed)
		random.seed(args.seed)
	torch.backends.cudnn.deterministic = reproducible
	torch.backends.cudnn.benchmark = not reproducible
	#Topology
	network = topology.create_graph(
		name=args.topology_name, size=args.nb_honests, 
		hyper=args.topology_hyper, byz=args.nb_real_byz,
		weights_method=args.topology_weights, seed=args.seed)	
	

	
	# Configurations
	config = experiments.Configuration(dtype=torch.float32, device=(None if args.device.lower() == "auto" else args.device), noblock=True)
	if args.device_gar.lower() == "same":
		config_gar = config
	else:
		config_gar = experiments.Configuration(dtype=config["dtype"], device=(None if args.device_gar.lower() == "auto" else args.device_gar), noblock=config["non_blocking"])
	# Defense
	if args.gars is None:
		defense = aggregators.gars.get(args.gar)
		if defense is None:
			tools.fatal_unavailable(aggregators.gars, args.gar, what="aggregation rule")
	else:
		def generate_defense(gars):
			# Preprocess given configuration
			freq_sum = 0.
			defenses = list()
			for info in gars.split(";"):
				# Parse GAR info
				info = info.split(",", maxsplit=2)
				name = info[0].strip()
				if len(info) >= 2:
					freq = info[1].strip()
					if freq == "-":
						freq = 1.
					else:
						freq = float(freq)
				else:
					freq = 1.
				if len(info) >= 3:
					try:
						conf = json.loads(info[2].strip())
						if not isinstance(conf, dict):
							tools.fatal(f"Invalid GAR arguments for GAR {name!r}: expected a dictionary, got {getattr(type(conf), '__qualname__', '<unknown>')!r}")
					except json.decoder.JSONDecodeError as err:
						tools.fatal(f"Invalid GAR arguments for GAR {name!r}: {str(err).lower()}")
				else:
					conf = dict()
				# Recover association GAR function
				defense = aggregators.gars.get(name)
				if defense is None:
					tools.fatal_unavailable(aggregators.gars, name, what="aggregation rule")
				# Store parsed defense
				freq_sum += freq
				defenses.append((defense, freq_sum, conf))
			# Return closure
			def unchecked(**kwargs):
				sel = random.random() * freq_sum
				for func, freq, conf in defenses:
					if sel < freq:
						return func.unchecked(**kwargs, **conf)
				return func.unchecked(**kwargs, **conf)  # Gracefully handle numeric imprecision
			def check(**kwargs):
				for defense, _, conf in defenses:
					message = defense.check(**kwargs, **conf)
					if message is not None:
						return message
			return aggregators.make_gar(unchecked, check)
		defense = generate_defense(args.gars)
		args.gar_args = dict()
	# Attack
	attack = attacks.attacks.get(args.attack)
	if attack is None:
		tools.fatal_unavailable(attacks.attacks, args.attack, what="attack")
	# No Model: there is only parameters.
	
	
	def loss(params, honest_params_init: torch.tensor):
		params_cp = torch.stack(params)
		target = honest_params_init.mean(dim=0)
		return ((params_cp - target)**2).sum(dim=0).mean()

	criterion = experiments.Criterion(args.criterion, **args.criterion_args)

	
	# Miscellaneous storage (step counter, momentum gradients, ...)
	storage = experiments.Storage()
	# Make the result directory (if requested)
	if args.result_directory is not None:
		try:
			resdir = pathlib.Path(args.result_directory).resolve()
			resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
			args.result_directory = resdir
		except Exception as err:
			tools.warning(f"Unable to create the result directory {str(resdir)!r} ({err}); no result will be stored")
		else:
			result_fds = dict()
			try:
				# Make evaluation file 
				if args.evaluation_delta > 0:
					result_make("eval", "Step number", "Average loss")
				# Make study file
				if args.nb_for_study > 0:
					result_make("study", "Step number", "Training point count",
						"Average loss", "l2 from origin", "Honest gradient deviation",
						"Honest gradient norm", "Honest max coordinate", "Honest Theta Deviation")
				# Store the configuration info and JSON representation
				(args.result_directory / "config").write_text(cmdline_config + os.linesep)
				with (args.result_directory / "config.json").open("w") as fd:
					def convert_to_supported_json_type(x):
						if type(x) in {str, int, float, bool, type(None), dict, list}:
							return x
						elif type(x) is set:
							return list(x)
						else:
							return str(x)
					datargs = dict((name, convert_to_supported_json_type(getattr(args, name))) for name in dir(args) if len(name) > 0 and name[0] != "_")
					del convert_to_supported_json_type
					json.dump(datargs, fd, ensure_ascii=False, indent="\t")
			except Exception as err:
				tools.warning(f"Unable to create some result files in directory {str(resdir)!r} ({err}); some result(s) may be missing")
	else:
		args.result_directory = None
		if args.checkpoint_delta != 0:
			args.checkpoint_delta = 0
			tools.warning("Argument '--checkpoint-delta' has been ignored as no '--result-directory' has been specified")

# ---------------------------------------------------------------------------- #
# Training
tools.success("Training ...")

class CheckConvertTensorError(RuntimeError):
	pass

def check_convert_tensor(tensor, refshape, config=config, errname="tensor"):
	""" Assert the given parameter is a tensor of the given reference shape, then convert it to the current config.
	Args:
		tensor   Tensor instance to assert
		refshape Reference shape to match
		config   Target configuration for the tensor
		errname  Name of what the tensor represents (only for the error messages)
	Returns:
		Asserted and converted tensor
	Raises:
		'CheckConvertTensorError' with explanatory message
	"""
	if not isinstance(tensor, torch.Tensor):
		raise CheckConvertTensorError(f"no/invalid {errname}")
	if tensor.shape != refshape:
		raise CheckConvertTensorError(f"{errname} has unexpected shape, expected {refshape}, got {tensor.shape}")
	try:
		return tensor.to(device=config["device"], dtype=config["dtype"], non_blocking=config["non_blocking"])
	except Exception as err:
		raise CheckConvertTensorError(f"converting/moving {errname} failed (err)")

storage["steps"] = 0
storage["datapoints"] = 0

# Training until limit or stopped
with tools.Context("training", "info"):
	steps_limit  = None if args.nb_steps < 0 else  args.nb_steps
	was_training = False
	current_lr   = None
	fd_eval    = result_get("eval")
	
	# initialize list of parameter vectors of honest workers
	d = args.nb_params
	
	honest_thetas_init = torch.randn((len(network),d))/(d**0.5)
	if args.random_init == 'normal_2':
		honest_thetas_init[:len(network)//2,:].add(torch.tensor([5] + [0]*(d-1)))
		honest_thetas_init[len(network)//2:,:].add(torch.tensor([5] + [0]*(d-1)))

    
    #check_convert_tensor(args.parameters_init, args.parameters_init.shape)
	honest_thetas = [torch.clone(honest_thetas_init[i]) for i in range(len(network))]
			
	while not exit_is_requested():
		steps    = storage["steps"]
		datapoints = storage["datapoints"]
		# ------------------------------------------------------------------------ #
		# Evaluate if any milestone is reached
		milestone_evaluation = args.evaluation_delta > 0 and steps % args.evaluation_delta == 0
		milestone_checkpoint = args.checkpoint_delta > 0 and steps % args.checkpoint_delta == 0
		milestone_user_input = args.user_input_delta > 0 and steps % args.user_input_delta == 0
		milestone_any    = milestone_evaluation or milestone_checkpoint or milestone_user_input
		# Training notification (end)
		if milestone_any and was_training:
			print(" done.")
			was_training = False

		# Evaluation milestone reached
		res_loss = 0
		nb_milestones = 0
		if milestone_evaluation:
			nb_milestones += 1
			print(f"loss (step {steps})...", end="", flush=True)
			res_loss = loss(honest_thetas, honest_thetas_init).item()

			#acc = (acc * (nb_milestones - 1) + acc_new) / nb_milestones
			print(f" loss {res_loss:.2f}.")
			# Store the evaluation result
			if fd_eval is not None:
				result_store(fd_eval, steps, res_loss)
		# Saving milestone reached
		if milestone_checkpoint:
			if args.load_checkpoint is None: # Avoid overwriting the checkpoint we just loaded
				filename = args.result_directory / f"checkpoint-{steps}" # Result directory is set and valid at this point
				print(f"Saving in {filename.name!r}...", end="", flush=True)
				try:
					experiments.Checkpoint().snapshot(model).snapshot(optimizer).snapshot(storage).save(filename, overwrite=True)
					print(" done.")
				except:
					tools.warning(" fail.")
					with tools.Context("traceback", "trace"):
						traceback.print_exc()
		args.load_checkpoint = None
		# User input milestone
		if milestone_user_input:
			tools.interactive()
		# Check if reach step limit
		if steps_limit is not None and steps >= steps_limit:
			# Training notification (end)
			if was_training:
				print(" done.")
				was_training = False
			# Leave training loop
			break
		# Training notification (begin)
		if milestone_any and not was_training:
			print("Training...", end="", flush=True)
			was_training = True
		
		# ------------------------------------------------------------------------ #
		## Sparse COMMUNICATION ##
		# ------------------------------------------------------------------------ #
		
		temp_thetas = []

		

		for honest_worker in network.nodes:	

			byz_thetas = attack.checked(
				grad_honests=honest_thetas, f_decl=args.nb_decl_byz, f_real=args.nb_real_byz, defense=defense,
				worker_attacked=honest_worker, network=network, factor=-16, clip_thresh=args.gradient_clip_centered, 
				gar=args.gar, bucket_size=args.bucket_size, gar_second = args.gar_second, current_step=steps,
				mimic_learning_phase=args.mimic_learning_phase, **args.attack_args)

			# Select the paramters and weights of neighboring nodes 
			
			indices = network.nodes 
			correct_thetas = [honest_thetas[k] for k in indices]
			
			# We compute the weights
			weights = network.weights(honest_worker)


			temp_thetas.append(
				defense.checked(gradients=(correct_thetas + byz_thetas), f=args.nb_decl_byz,
                honest_index=honest_worker, weights=weights, communication_step = network.communication_step, 
				clip_thresh=args.gradient_clip_centered, bucket_size=args.bucket_size, gar_second = args.gar_second, current_step=steps,
                mimic_learning_phase=args.mimic_learning_phase, **args.gar_args)
				)
			

		# Update the previous (honest) thetas
		previous_thetas = honest_thetas
		# Update the honest_thetas
		honest_thetas = temp_thetas

		# ------------------------------------------------------------------------ #
		# Increase the step counter
		storage["steps"]    = steps + 1
		storage["datapoints"] = datapoints + 1
	# Training notification (end)
	if was_training:
		print(" interrupted.")
