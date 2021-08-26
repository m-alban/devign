import java.io.File
import scala.io.Source
import java.io.PrintWriter

def export_cpg(file_path: String) = {
  // setup for joern
  val src_string = Source.fromFile(file_path).getLines.mkString
  importCode.c.fromString(src_string)
  run.ossdataflow
  val ast_ids = cpg.method.l(0).ast.map(n=>n.id).l
  val cfg_edges = 
    (cpg.method.l(0).graph.edges
      .filter(e => (e.label).equalsIgnoreCase("CFG"))
      // make sure the edge is within the AST
      .filter(e => ast_ids.contains(e.inNode.id) & ast_ids.contains(e.outNode.id))
      .map(e => e.bothNodes.map(n=>n.id).l).l)
  val ast_edges = 
    (cpg.method.l(0).graph.edges
      .filter(e => (e.label).equalsIgnoreCase("AST"))
      .filter(e => ast_ids.contains(e.inNode.id) & ast_ids.contains(e.outNode.id))  
      .map(e => e.bothNodes.map(n=>n.id).l).l)
  // creating the strings of edge lists to be stored in json for python
  val cfg_edges_str = 
    ("[" 
      + cfg_edges.map(l => "[" + l(0).toString + "," + l(1).toString + "]").mkString(",")
      +"]")
  val ast_edges_str = 
    ("[" 
      + ast_edges.map(l => "[" + l(0).toString + "," + l(1).toString + "]").mkString(",")
      +"]")
  val ast_json = cpg.method.l(0).ast.toJson
  val out_str = 
    ("{\"ast_nodes\":" + ast_json 
      + ", \"ast_edges\":" + ast_edges_str 
      + ", \"cfg_edges\":" + cfg_edges_str 
      + "}")
  // get the source file name for naming the graph json file
  val fp_array = file_path.split("/")
  val file_name = fp_array(fp_array.size - 1)
  val out_file_name = file_name.substring(0, file_name.length - 1) + "json"
  val out_path_list = 
    (fp_array.slice(0, fp_array.size-2) 
      :+ "graphs"
      :+ out_file_name)
  val out_path = out_path_list.mkString("/")
  new PrintWriter(out_path) {try {write(out_str)} finally {close}}
  close
}

def export_cpg_wrapper(file_path: String) = {
  try {
    export_cpg(file_path)
  } catch {
    case e: Throwable => log_error(file_path, "joern error:\n        " + e)
  }
}

def log_error(file_path: String, err_message: String) = {
  val split_path = file_path.split('/')
  val out_path_base = split_path.slice(0, split_path.length - 3).mkString("/")
  val err_path = out_path_base + "/log/graph_export.log"
  var contents = ""
  if(new File(err_path).exists){
    contents = Source.fromFile(err_path).mkString
  }
  contents += (file_path+'\n'+"    "+err_message+ '\n')
  new PrintWriter(err_path) {try {write(contents)} finally {close}}
}

@main def process_dir(src_file_dir: String) = {
  val dir = new File(src_file_dir)
  dir.listFiles.toList.map(_.toString)
                      .foreach(file_path=>export_cpg_wrapper(file_path))
}
