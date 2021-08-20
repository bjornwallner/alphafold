import os
import pickle
import sys
import time
import pathlib

from absl import app
from absl import flags
from absl import logging

from alphafold.data import pipeline

#from alphafold.common import protein
#from alphafold.data import pipeline
#from alphafold.data import parsers
from alphafold.data import templates
#from alphafold.model import data
#from alphafold.model import config
#from alphafold.model import model
#from alphafold.relax import relax

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

#jackhmmer_binary_path='/proj/wallner/apps/hmmer-3.2.1/bin/jackhmmer'
#hhblits_binary_path='/proj/wallner/apps/hhsuite/bin/hhblits'
#hhsearch_binary_path= '/proj/wallner/apps/hhsuite/bin/hhsearch'                  
#kalign_binary_path='/proj/wallner/apps/kalign/src/kalign'

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
MAX_TEMPLATE_HITS = 20
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print(FLAGS.preset)

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


  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  output_dir_base=FLAGS.output_dir
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    logging.info(f'Running: {fasta_path}')
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)
    features_output_path = os.path.join(output_dir, 'features.pkl')
    if not os.path.exists(features_output_path):
        feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path,
            msa_output_dir=msa_output_dir)
        # Write out features as a pickled dictionary.
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
#      'model_names',
#      'data_dir',
#      'preset',
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
    
