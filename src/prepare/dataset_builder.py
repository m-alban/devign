from ..utils import PROJECT_ROOT

import gensim
import json
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pickle
import re
import shutil
import subprocess
import torch
import torch_geometric.data as geom_data
from typing import Dict, List

class TorchGraphPrepError(Exception):
    """ Error in preparing data for a torch graph
    """
    pass

class JoernExeError(Exception):
    """ Error in running Joern
    """
    pass

class DatasetBuilder:
    """A class to set up the project dataset. 

    Use this to generate or reuse the dataset source files and json graphs, and
        to build the torch data.

    Attrributes:
        embedding_size: the size of the embedding for a single node in a graph
        label_encode: Dict[str, numpy.array] mapping AST labels to their embedding
        project_dir: Path of the base directory for the project
        tokenizer: RegexpTokenizer tokenize C source code
        word_vectors: gensim Word2VecKeyedVectors, encode word vectors for source code

    """
    def __init__(self, fresh_build: bool, test_build: bool) -> None:
        """Set up the builder
        
        Args:
            fresh_build: True to generate dataset source files and graphs,
                deleting any existing. False to reuse existing files.
            test_build: True to build on a sample of the dataset.
            
        Raises:
            JoernExeError: if running Joern fails.
        """
        c_regexp = r'\w+|->|\+\+|--|<=|>=|==|!=|' + \
                   r'<<|>>|&&|\|\||-=|\+=|\*=|/=|%=|' + \
                   r'&=|<<=|>>=|^=|\|=|::|' + \
                   r"""[!@#$%^&*()_+-=\[\]{};':"\|,.<>/?]"""
        self.tokenizer = RegexpTokenizer(c_regexp)
        if test_build:
            self.project_dir = os.path.join(PROJECT_ROOT, 'sample')
        else:
            self.project_dir = str(PROJECT_ROOT)
        if fresh_build:
            self.cleanup()
            self.create_src_files()
            joern_check_path = os.path.join(PROJECT_ROOT, 'joern', 'joern')
            if os.path.exists(joern_check_path):
                joern_dir = joern_check_path
            else:
                joern_dir = os.path.join(os.path.expanduser('~'), 'bin', 'joern')
            # execute joern work
            joern_exe = os.path.join(joern_dir, 'joern-cli', 'joern')
            src_file_dir = os.path.join(self.project_dir, 'data', 'src_files')
            script_path = os.path.join(PROJECT_ROOT, 'joern', 'export_cpg.sc')
            joern_command = f'{joern_exe} --script {script_path} --params src_file_dir="{src_file_dir}"'
            res = subprocess.call(joern_command, shell=True, 
                                  cwd = self.project_dir,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            if res != 0:
                raise JoernExeError 
            # build out embedding
            self.encode_labels()
            self.train_word2vec()
        # models into object attributes
        model_dir = os.path.join(self.project_dir, 'models')
        word_vector_path = os.path.join(model_dir, 'src_code.wordvectors')
        self.word_vectors = gensim.models.KeyedVectors.load(word_vector_path, mmap='r')
        label_set_path = os.path.join(model_dir, 'label_set.pickle')
        with open(label_set_path, 'rb') as label_f:
            label_set = pickle.load(label_f)
        labels = sorted(list(label_set))
        # OHE in this manner b/c doing one sample at a time
        self.label_encode = {}
        for i in range(len(labels)):
            vec = np.zeros(len(labels))
            vec[i] = 1.
            self.label_encode[labels[i]] = vec
        self.embedding_size = self.word_vectors.vector_size + len(labels)

    def build_graphs(self, num_nodes: int) -> List[geom_data.Data]:
        """Get a list of torch graphs for the project's dataset

        Returns:
            A list of torch graph objects
        """
        #setup data locations
        graph_dir = os.path.join(self.project_dir, 'data', 'graphs')
        [(_, _, graph_name_list)] = [x for x in os.walk(graph_dir)]
        #setup containers
        dataset = []
        graph_build_err_list = []
        #setup encoders
        for f_name in graph_name_list:
            try:
                torch_graph = self.prepare_torch_data(f_name, num_nodes)
            except TorchGraphPrepError:    
                graph_build_err_list.append(f_name)
                continue
            dataset.append(torch_graph)
        log_path = os.path.join(self.project_dir, "log", "graph_build_err.log")
        with open(log_path, 'w') as log_f:
            for f_name in graph_build_err_list:
                log_f.write(f_name + '\n')
        return dataset

    def cleanup(self) -> None:
        """Clean up the project directory, removing generated files/data

        Requires:
            self.project_dir has been assigned
        """
        def clean(dir_path):
            [(_, _, files)] = [x for x in os.walk(dir_path)]
            for f in files: os.remove(os.path.join(dir_path, f))
        data_dir = os.path.join(self.project_dir, 'data')
        src_file_dir = os.path.join(data_dir, 'src_files')
        graph_dir = os.path.join(data_dir, 'graphs')
        log_dir = os.path.join(self.project_dir, 'log')
        model_dir = os.path.join(self.project_dir, 'models')
        #find all subdirectories to the project that are the workspace generated by joern
        workspace_dir = os.path.join(self.project_dir, 'workspace')
        if os.path.isdir(workspace_dir):
            shutil.rmtree(workspace_dir)
        cleaned_dir_list = [src_file_dir, graph_dir, log_dir, model_dir]
        for dir_path in cleaned_dir_list: clean(dir_path)

    def create_src_files(self) -> None:
        """Creates source files of raw dataset. File name is <unique id>+<target label>
        """
        data_dir = os.path.join(self.project_dir, "data")
        dataset_path = os.path.join(data_dir, "dataset.json")
        src_file_dir = os.path.join(data_dir, "src_files")
        with open(dataset_path, 'r') as dataset_f:
            dataset = json.load(dataset_f)
            i = 0 # index for unique observation key
            for i, sample in enumerate(dataset):
                file_name = f"{i}_{sample['target']}.c"
                with open(os.path.join(src_file_dir, file_name), 'w') as write_f:
                    write_f.write(sample["func"])

    def encode_labels(self) -> None:
        """Save the label encoding for AST node labels
        """
        graph_dir = os.path.join(self.project_dir, "data", "graphs")
        [(_, _, graph_name_list)] = [x for x in os.walk(graph_dir)]
        label_set = set({})
        for f_name in graph_name_list:
            with open(os.path.join(graph_dir, f_name)) as f:
                json_data = json.load(f)
                ast_nodes = json_data['ast_nodes']
                for node in ast_nodes:
                    label_set.add(node["_label"])
        model_dir = os.path.join(self.project_dir,  "models")
        with open(os.path.join(model_dir, "label_set.pickle"), "wb") as label_f:
            pickle.dump(label_set, label_f)

    def get_target(self, file_name: str) -> int:
        """Get the target given a file name
        """
        target_str = file_name.split("_")[1]
        non_num = re.compile(r'[^\d]')
        target = int(non_num.sub('', target_str))
        return target

    def prepare_torch_data(self, file_name:str, num_nodes: int) -> geom_data.Data:
        """Prepares the ast edges and nodes for constructing torch graph

        Args:
            file_name: the name of graph file
            num_nodes: The number of nodes to be used for each graph. Those with more/less
                will be truncated/0 padded.

        Raises:
            TorchGraphPrepError: If there is an error finding node IDs

        Returns:
            torch_geometric graph representing the ast
        """
        graph_path = os.path.join(self.project_dir, 'data', 'graphs', file_name)
        with open(graph_path) as f:
            json_data = json.load(f)
        ast_nodes = json_data['ast_nodes']
        # map Joern ID to embedding
        node_embedding_dict = {}
        # embed nodes, map node id to embedding
        for n in ast_nodes:
            if 'code' in n:
                n_code = n['code']
            else:
                n_code = ""
            try: #TODO shrink try block
                n_code = n_code.replace('\\t', ' ')
                if not n_code:
                    code_embedding = np.zeros(self.word_vectors.vector_size)
                else:
                    code_embedding = np.mean([self.word_vectors[x] 
                                              for x in self.tokenizer.tokenize(n_code)], axis=0)
                label_embedding = self.label_encode[n["_label"]]
                node_embedding = np.append(code_embedding, label_embedding)
                node_embedding_dict[n['id']] = node_embedding
            except KeyError:
                raise TorchGraphPrepError
        old_to_new_id_dict = {}
        # hold node embeddings in order
        if num_nodes == np.inf:
            num_embeddings = len(node_embedding_dict)
        else:
            num_embeddings = num_nodes
        embeddings = [np.zeros(self.embedding_size)]*num_embeddings
        # truncate the graph to include correct number of nodes
        truncated_graph_id_list = sorted(node_embedding_dict)[:len(embeddings)]
        for new_id, old_id in enumerate(truncated_graph_id_list):
            old_to_new_id_dict[old_id] = new_id
            embeddings[new_id] = node_embedding_dict[old_id]
        #reset edges according to new node ids
        new_ast_edges = []
        for edge in json_data["ast_edges"]:
            if edge[0] in old_to_new_id_dict and edge[1] in old_to_new_id_dict:
                try:
                    new_ast_edges.append([old_to_new_id_dict[n] for n in edge])
                except KeyError:
                    raise TorchGraphPrepError
        edge_index = torch.tensor(new_ast_edges, dtype=torch.int64)
        x = torch.tensor(embeddings, dtype=torch.float)
        y = torch.tensor([self.get_target(file_name)], dtype=torch.long)
        data = geom_data.Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
        return data

    def train_word2vec(self) -> None:
        """Train word2vec model on all source code.
        """
        src_file_dir = os.path.join(self.project_dir, "data", "src_files")
        [(_, _, src_name_list)] = [x for x in os.walk(src_file_dir)] 
        corpus = []
        for f_name in src_name_list:
            with open(os.path.join(src_file_dir, f_name)) as f:
                lines = f.readlines()
                corpus += lines
        corpus = [self.tokenizer.tokenize(line) for line in corpus]
        corpus = [x for x in corpus if len(x) > 0]
        model = gensim.models.Word2Vec(corpus, size=100, iter=10, min_count=1)
        word_vectors = model.wv
        model_dir = os.path.join(self.project_dir, "models")
        word_vectors.save(os.path.join(model_dir, "src_code.wordvectors"))
