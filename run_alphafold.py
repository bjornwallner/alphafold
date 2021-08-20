# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Dict
from string import ascii_uppercase

from absl import app
from absl import flags
from absl import logging
import numpy as np

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax

# Internal import (7716).




# Path to directory of supporting data, contains 'params' dir.
data_dir = '/proj/wallner/share/alphafold_data'
DOWNLOAD_DIR= data_dir

# Path to the Uniref90 database for use by JackHMMER.
uniref90_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniref90', 'uniref90.fasta')

# Path to the MGnify database for use by JackHMMER.
mgnify_database_path = os.path.join(
    DOWNLOAD_DIR, 'mgnify', 'mgy_clusters.fa')

# Path to the BFD database for use by HHblits.
bfd_database_path = os.path.join(
    DOWNLOAD_DIR, 'bfd',
    'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')

# Path to the Uniclust30 database for use by HHblits.
#uniclust30_database_path = os.path.join(
#    DOWNLOAD_DIR, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')

uniclust30_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniclust30', 'UniRef30_2021_06', 'UniRef30_2021_06')

# Path to the PDB70 database for use by HHsearch.
pdb70_database_path = os.path.join(DOWNLOAD_DIR, 'pdb70', 'pdb70')

# Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
template_mmcif_dir = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'mmcif_files')

# Path to a file mapping obsolete PDB IDs to their replacements.
obsolete_pdbs_path = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'obsolete.dat')

#### END OF USER CONFIGURATION ####

# Names of models to use.
model_names = [
    'model_1',
    'model_2',
    'model_3',
    'model_4',
    'model_5',
]



flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_list('model_names', model_names, 'Names of models to use.')
flags.DEFINE_string('data_dir', data_dir, 'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', '/proj/wallner/apps/hmmer-3.2.1/bin/jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', '/proj/wallner/apps/hhsuite/bin/hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', '/proj/wallner/apps/hhsuite/bin/hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', '/proj/wallner/apps/kalign/src/kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', uniref90_database_path, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', mgnify_database_path, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', bfd_database_path, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('uniclust30_database_path', uniclust30_database_path, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', pdb70_database_path, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', template_mmcif_dir, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', '2050-01-01', 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', obsolete_pdbs_path, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('preset', 'full_dbs', ['full_dbs', 'casp14'],
                  'Choose preset model configuration - no ensembling '
                  '(full_dbs) or 8 model ensemblings (casp14).')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_boolean('exit_after_sequence_search',False,'Will exit after sequence search')
flags.DEFINE_boolean('skip_bfd',False,'Skip the large BFD database (1.5TB) search')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('nstruct',1,'number of structures to generate for each model')
flags.DEFINE_integer('chainbreak_offset',200,'number to offset the residue index in case of chain break')


FLAGS = flags.FLAGS
#print(FLAGS)
#sys.exit()
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int):
  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  #check fasta file for chainbreaks
  with open(fasta_path) as f:
    input_fasta_str = f.read()
  input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
  if len(input_seqs) != 1:
    raise ValueError(f'More than one input sequence found in {input_fasta_path}.')
  input_sequence = input_seqs[0]
  input_description = input_descs[0]

  chain_numbering=[]
  seqs=input_sequence.split('/')
  fasta_concat=os.path.join(output_dir,f'concat.fasta')
  with open(fasta_concat,'w') as f:
    f.write(f'>{input_description} concat\n')
    f.write("".join(seqs))
    f.write('\n')
  number_of_chains=0
  for n,seq in enumerate(seqs):
    chain=ascii_uppercase[n]
    number_of_chains+=1
    fasta_out=os.path.join(output_dir,f'chain{chain}.fasta')
    #print(fasta_out)
    with open(fasta_out,'w') as f:
        f.write(f'>{input_description} chain {chain}\n')
        for s in seq:
          f.write(s)
          chain_numbering.append(n+1)
        f.write('\n')

  #for a,b in zip("".join(seqs),chain_numbering):
  #  print(a,b)
    
  print(input_sequence)
  print(number_of_chains)



  #Currently an MSA is made for the concatinated sequence
  #Future feature will be to do the MSAs separately an merge them like so or paired:
   # make multiple copies of msa for each copy
  # AAA------
  # ---AAA---
  # ------AAA
  #
  # note: if you concat the sequences (as below), it does NOT work according to https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)
  # AAAAAAAAA
  
  fasta_path=fasta_concat
  # Get features.
  t_0 = time.time()
  features_output_path = os.path.join(output_dir, 'features.pkl')
  if not os.path.exists(features_output_path):
    feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir)

    # Write out features as a pickled dictionary.
    
  
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  else:
    feature_dict=pickle.load(open(features_output_path,'rb'))
  if FLAGS.exit_after_sequence_search:
    sys.exit()
  if number_of_chains > 1:
   # Based on Minkyung's code
   # add big enough number to residue index to indicate chain breaks
    #pointer to feature_dict
    idx_res = feature_dict['residue_index']
    for i,_ in enumerate(idx_res):
      idx_res[i] += FLAGS.chainbreak_offset*chain_numbering[i]
    #chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
    #feature_dict['residue_index'] = idx_res
    
      # Write out modified features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, 'features.modified.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)

  timings['features'] = time.time() - t_0
  relaxed_pdbs = {}
  plddts = {}

  # Run the models.
  for network_model_name, model_runner in model_runners.items():
    logging.info('Running model %s', network_model_name)
    t_0 = time.time()
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)
    timings[f'process_features_{network_model_name}'] = time.time() - t_0

    t_0 = time.time()
    for model_no in range(1,FLAGS.nstruct+1):
      model_name=f'{network_model_name}_{model_no}'
      unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
      relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
      if not os.path.exists(relaxed_output_path):
      
        prediction_result = model_runner.predict(processed_feature_dict)
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
          'Total JAX model %s predict time (includes compilation time, see --benchmark): %.0f?',
          model_name, t_diff)

        if benchmark:
          t_0 = time.time()
          model_runner.predict(processed_feature_dict)
          timings[f'predict_benchmark_{model_name}'] = time.time() - t_0

        # Get mean pLDDT confidence metric.
        plddts[model_name] = np.mean(prediction_result['plddt'])

        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)
        unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                prediction_result)
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))
          f.write(f"pLLDT MEAN   {np.mean(prediction_result['plddt'])}\n")
          f.write(f"pLLDT MEDIAN {np.median(prediction_result['plddt'])}\n")
        with open(unrelaxed_pdb_path+'.plldt', 'w') as f:
          for pos,plddt in enumerate(prediction_result['plddt'],1):
            f.write(f'{pos} {plddt}\n')
        
          

      
      # Relax the prediction.

#      if not os.path.exists(relaxed_output_path):
        t_0 = time.time()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)
                  

      # Rank by pLDDT and write out relaxed PDBs in rank order.
      ranked_order = []
      for idx, (model_name, _) in enumerate(
          sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
          f.write(relaxed_pdbs[model_name])
      
      ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
      with open(ranking_output_path, 'w') as f:
        f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))
      
      logging.info('Final timings for %s: %s', fasta_name, timings)
      
      timings_output_path = os.path.join(output_dir, 'timings.json')
      with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print(FLAGS.preset)

  if FLAGS.preset == 'full_dbs':
    num_ensemble = 1
  elif FLAGS.preset == 'casp14':
    num_ensemble = 8

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      pdb70_database_path=FLAGS.pdb70_database_path,
      template_featurizer=template_featurizer,
      skip_bfd=FLAGS.skip_bfd)
 
  model_runners = {}
  for model_name in FLAGS.model_names:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
#      'model_names',
#      'data_dir',
      'preset',
#      'uniref90_database_path',
#      'mgnify_database_path',
#      'uniclust30_database_path',
#      'bfd_database_path',
#      'pdb70_database_path',
#      'template_mmcif_dir',
#      'max_template_date',
#      'obsolete_pdbs_path',
  ])

  app.run(mai
