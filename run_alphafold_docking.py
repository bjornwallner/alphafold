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
from alphafold.common import residue_constants
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



flags.DEFINE_string('fasta1', None, 'Paths to FASTA file 1')
flags.DEFINE_string('fasta2', None, 'Paths to FASTA file 2')
flags.DEFINE_string('msa1', None, 'MSA1.')
flags.DEFINE_string('msa2', None, 'MSA2')


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
flags.DEFINE_integer('msas_to_use',3,'number of msa methods to use')
flags.DEFINE_integer('max_recycles', 3,     'Number of recyles through the model')                                          
flags.DEFINE_integer('tolerance', 0,     'Minimal CA RMS between recycles to keep recycling')                            

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
      result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
      prediction_result={}
      unrelaxed_protein=None
      if os.path.exists(unrelaxed_pdb_path):
        logging.info(f'Found {unrelaxed_pdb_path}... loading predictions...')
        with open(result_output_path,'rb') as f:
          prediction_result=pickle.load(f)
          plddt = prediction_result['plddt']
          plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
          unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors)
      else:
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
        plddt = prediction_result['plddt']
        plddts[model_name] = np.mean(plddt)
        # Save the model outputs.

        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)

        
#        unrelaxed_protein = protein.from_prediction(processed_feature_dict,
#                                                prediction_result)
         # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
          plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors)
        
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))
          f.write(f"pLLDT MEAN   {np.mean(prediction_result['plddt'])}\n")
          f.write(f"pLLDT MEDIAN {np.median(prediction_result['plddt'])}\n")
        with open(unrelaxed_pdb_path+'.plldt', 'w') as f:
          for pos,plddt in enumerate(prediction_result['plddt'],1):
            f.write(f'{pos} {plddt}\n')
  
      
      # Relax the prediction.
      if not os.path.exists(relaxed_output_path):
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

def paste_msa(msa1,dmsa1,msa2,dmsa2):
# AAA---
# ---BBB
  L1=len(msa1[0])
  L2=len(msa2[0])
  msas = []
  deletion_matrices = []
  msas=[seq + '-'*L2 for seq in msa1] + ['-'*L1 + seq for seq in msa2]
  deletion_matrices=[mtx+[0]*L2 for mtx in dmsa1] + [[0]*L1+mtx for mtx in dmsa2]
  
  return(msas,deletion_matrices)

  


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  target_name=f"{os.path.basename(FLAGS.fasta1)}-{os.path.basename(FLAGS.fasta2)}".replace('.fa','')
  
  num_ensemble = 1
  output_dir_base=FLAGS.output_dir
  output_dir = os.path.join(output_dir_base, target_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  input_seqs1, input_descs1 = parsers.parse_fasta(open(FLAGS.fasta1,'r').read())
  input_sequence1 = input_seqs1[0]
  input_description1 = input_descs1[0]

  input_seqs2, input_descs2 = parsers.parse_fasta(open(FLAGS.fasta2,'r').read())
  input_sequence2 = input_seqs2[0]
  input_description2 = input_descs2[0]
  input_description=input_description1+input_description2
  input_sequence=input_sequence1+input_sequence2
  num_res = len(input_sequence)
  num_res1=len(input_sequence1)
  mgnify_max_hits=501

  

  msas=[]
  deletion_matrices=[]

  uniref90_file1=os.path.join(FLAGS.msa1,'uniref90_hits.sto')
  uniref90_msa1, uniref90_deletion_matrix1 = parsers.parse_stockholm(open(uniref90_file1,'r').read())
  uniref90_file2=os.path.join(FLAGS.msa2,'uniref90_hits.sto')
  uniref90_msa2, uniref90_deletion_matrix2 = parsers.parse_stockholm(open(uniref90_file2,'r').read())
  uniref90_msa,uniref90_deletion_matrix = paste_msa(uniref90_msa1, uniref90_deletion_matrix1,uniref90_msa2, uniref90_deletion_matrix2)
  msas.append(uniref90_msa)
  deletion_matrices.append(uniref90_deletion_matrix)
  uniref90_msa1.clear()
  uniref90_msa2.clear()
  uniref90_deletion_matrix1.clear()
  uniref90_deletion_matrix2.clear()

  if FLAGS.msas_to_use > 1:
    bfd_file1=os.path.join(FLAGS.msa1,'bfd_uniclust_hits.a3m')
    bfd_msa1, bfd_deletion_matrix1 = parsers.parse_a3m(open(bfd_file1,'r').read())
    bfd_file2=os.path.join(FLAGS.msa2,'bfd_uniclust_hits.a3m')
    bfd_msa2, bfd_deletion_matrix2 = parsers.parse_a3m(open(bfd_file2,'r').read())
    bfd_msa, bfd_deletion_matrix = paste_msa(bfd_msa1, bfd_deletion_matrix1, bfd_msa2, bfd_deletion_matrix2)
    msas.append(bfd_msa)
    deletion_matrices.append(bfd_deletion_matrix)
    bfd_msa1.clear()
    bfd_msa2.clear()
    bfd_deletion_matrix1.clear()
    bfd_deletion_matrix2.clear()


  if FLAGS.msas_to_use > 2:
    mgnify_file1=os.path.join(FLAGS.msa1,'mgnify_hits.sto')
    mgnify_msa1, mgnify_deletion_matrix1 = parsers.parse_stockholm(open(mgnify_file1,'r').read())
    mgnify_msa1 = mgnify_msa1[:mgnify_max_hits]
    mgnify_deletion_matrix1 = mgnify_deletion_matrix1[:mgnify_max_hits]

    mgnify_file2=os.path.join(FLAGS.msa2,'mgnify_hits.sto')
    mgnify_msa2, mgnify_deletion_matrix2 = parsers.parse_stockholm(open(mgnify_file2,'r').read())
    mgnify_msa2 = mgnify_msa2[:mgnify_max_hits]
    mgnify_deletion_matrix2 = mgnify_deletion_matrix2[:mgnify_max_hits]
    mgnify_msa, mgnify_deletion_matrix = paste_msa(mgnify_msa1, mgnify_deletion_matrix1,mgnify_msa2, mgnify_deletion_matrix2)

    #msas=(uniref90_msa,bfd_msa,mgnify_msa)
    #deletion_matrices=(uniref90_deletion_matrix,bfd_deletion_matrix,mgnify_deletion_matrix)
    msas.append(mgnify_msa)
    deletion_matrices.append(mgnify_deletion_matrix)

#  hhrfile=os.path.join(FLAGS.msa1,'hhsearch_uniref_max_hits10000.hhr')





  

#  hhsearch_hits = parsers.parse_hhr(open(hhrfile,'r').read())


  hhsearch_hits=''
  #print(msas)
  #print(len(uniref90_msa1))
  #print(len(uniref90_msa1[0]))

  #print(len(uniref90_msa2))
  #print(len(uniref90_msa2[0]))
  

  #print("===============")
  #print(num_res)
  #print(len(msas))
  #print(len(msas[0]))
  #print(len(msas[0][0]))
 #seen_sequences = set()
 #for msa_index, msa in enumerate(msas):
 #  for sequence_index, sequence in enumerate(msa):
 #    if sequence in seen_sequences:
 #      continue
 #    seen_sequences.add(sequence)
 #    
 #
 #sys.exit()
  #print(hhsearch_hits)
  hhsearch_hits=''
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
 
 
  model_runners = {}
  for model_name in FLAGS.model_names:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = FLAGS.max_recycles 
    model_config.model.num_recycle = FLAGS.max_recycles
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

  template_results = template_featurizer.get_templates(
    query_sequence=input_sequence,
    query_pdb_code=None,
    query_release_date=None,
    hhr_hits=hhsearch_hits)
  feature_dict = {
    **pipeline.make_sequence_features(
      sequence=input_sequence,
      description=input_description,
      num_res=num_res),
    **pipeline.make_msa_features(
      msas=msas, 
      deletion_matrices=deletion_matrices),
    **template_results.features
  }

  for n, msa in enumerate(msas):         
    logging.info('MSA %d size: %d sequences.', n, len(msa))     
    logging.info('Final (deduplicated) MSA size: %d sequences.',feature_dict['num_alignments'][0])
    logging.info('Total number of templates: %d.', template_results.features['template_domain_names'].shape[0]) 
    
#  sys.exit()
  features_output_path = os.path.join(output_dir, f'features_msas{FLAGS.msas_to_use}.pkl')
  with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
    

      # Based on Minkyug's code
      # add big enough number to residue index to indicate chain breaks
      #pointer to feature_dict
  idx_res = feature_dict['residue_index']
  for i,_ in enumerate(feature_dict['residue_index']):
    if i>=num_res1:
      feature_dict['residue_index'][i] += FLAGS.chainbreak_offset
    print(i, input_sequence[i], feature_dict['residue_index'][i])
        #idx_res[i] += FLAGS.chainbreak_offset
    #chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
    #feature_dict['residue_index'] = idx_res
    





      # Write out modified features as a pickled dictionary.
    #features_output_path = os.path.join(output_dir, 'features.modified.pkl')
    features_output_path = os.path.join(output_dir, f'features_msas{FLAGS.msas_to_use}_chainbreak_offset{FLAGS.chainbreak_offset}.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  unrelaxed_pdbs = {}
  plddts = {}
  # Run the models.
  for network_model_name, model_runner in model_runners.items():
    logging.info('Running model %s', network_model_name)
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)
    for model_no in range(1,FLAGS.nstruct+1):
      model_name=f'{network_model_name}_msas{FLAGS.msas_to_use}_chainbreak_offset{FLAGS.chainbreak_offset}_recycles{FLAGS.max_recycles}_{model_no}'
      unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
      relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
      result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
      prediction_result={}
      unrelaxed_protein=None
      if os.path.exists(unrelaxed_pdb_path):
        logging.info(f'Found {unrelaxed_pdb_path}... loading predictions...')
        with open(result_output_path,'rb') as f:
          prediction_result=pickle.load(f)
          plddt = prediction_result['plddt']
          plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
          unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors)
      else:
        prediction_result = model_runner.predict(processed_feature_dict)
        # Get mean pLDDT confidence metric.
        plddt = prediction_result['plddt']
        plddts[model_name] = np.mean(plddt)
        # Save the model outputs.

        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)

        
#        unrelaxed_protein = protein.from_prediction(processed_feature_dict,
#                                                prediction_result)
         # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
          plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors)
        
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))
          f.write(f"pLLDT MEAN   {np.mean(prediction_result['plddt'])}\n")
          f.write(f"pLLDT MEDIAN {np.median(prediction_result['plddt'])}\n")
        with open(unrelaxed_pdb_path+'.plldt', 'w') as f:
          for pos,plddt in enumerate(prediction_result['plddt'],1):
            f.write(f'{pos} {plddt}\n')
        with open(unrelaxed_pdb_path, 'r') as f:
          unrelaxed_pdbs[model_name]=f.read()
      
      # Rank by pLDDT and write out relaxed PDBs in rank order.
      ranked_order = []
      for idx, (model_name, _) in enumerate(
          sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
          f.write(unrelaxed_pdbs[model_name])
      
      ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
      with open(ranking_output_path, 'w') as f:
        f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))
      

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta1',
    'fasta2',
    'msa1',
    'msa2',
      'output_dir',
#      'model_names',
#      'data_dir',
#     'preset',
#      'uniref90_database_path',
#      'mgnify_database_path',
#      'uniclust30_database_path',
#      'bfd_database_path',
#      'pdb70_database_path',
#      'template_mmcif_dir',
#      'max_template_date',
#      'obsolete_pdbs_path',
  ])

  app.run(main)
  
