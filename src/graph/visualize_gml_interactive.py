
# Locate the package from this file, not from the caller's working directory.
from pathlib import Path as _PortablePath
import sys as _portable_sys
_PORTABLE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents
                      if (p / "featlens_paths.py").is_file())
if str(_PORTABLE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_ROOT))
from featlens_paths import (package_path as _package_path,
                            external_path as _external_path,
                            model_location as _model_location,
                            evaluation_python as _evaluation_python,
                            temporary_path as _temporary_path)

from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import networkx as nx
from pyvis.network import Network


FUNCTION_COLOR = "#99CCFF"
CLASS_COLOR = "#F8D8B2"
DEFAULT_COLOR = "#D9D9D9"
HOST = "0.0.0.0"
PORT = 5090

# Edit the GML file path to visualize directly here.
GML_FILE_PATH = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/graphs/task_17_ori.gml'))
# GML_FILE_PATH = "<package_root>/results/repoSummaryOut/DevEval/System/mrjob/graph_results_all/task_17_mid.gml"
# GML_FILE_PATH = "<package_root>/results/repoSummaryOut/DevEval/System/mrjob/graph_results_all/PageRank-15-subgraph/task_17_rank.gml"


def format_sig_label(sig: str, max_line_len: int = 24, max_lines: int = 3) -> str:
    parts = [p for p in sig.split(".") if p]
    if not parts:
        return sig

    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}.{part}"
        if len(candidate) <= max_line_len:
            current = candidate
            continue

        if current:
            lines.append(current)
        else:
            lines.append(part[: max_line_len - 1] + "…")
            current = ""
            if len(lines) >= max_lines:
                break
            continue

        current = part
        if len(lines) >= max_lines:
            break

    if len(lines) < max_lines and current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    if len(lines) == max_lines and ".".join(parts) != ".".join(lines):
        lines[-1] = (lines[-1][: max_line_len - 1] + "…") if len(lines[-1]) >= max_line_len else (lines[-1] + "…")

    return "\n".join(lines)


def choose_node_color(node_data: dict) -> str:
    category = str(node_data.get("category", node_data.get("catogory", ""))).strip().lower()
    if category == "function":
        return FUNCTION_COLOR
    if category == "class":
        return CLASS_COLOR
    return DEFAULT_COLOR


def infer_edge_type(edge_data: dict) -> str:
    for key in ("type", "kind", "relation", "rel", "label"):
        value = edge_data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "Unknown"


def build_network(gml_path: Path) -> Network:
    graph = nx.read_gml(gml_path)
    is_directed = graph.is_directed()

    net = Network(
        height="900px",
        width="100%",
        directed=is_directed,
        bgcolor="#FFFFFF",
        font_color="#222222",
        cdn_resources="remote",
    )

    for node_id, data in graph.nodes(data=True):
        sig = str(data.get("sig", str(node_id)))
        label_sig = format_sig_label(sig)
        category = str(data.get("category", data.get("catogory", "Unknown")))
        title = f"<b>id:</b> {node_id}<br><b>sig:</b> {sig}<br><b>category:</b> {category}"

        net.add_node(
            n_id=str(node_id),
            label=label_sig,
            title=title,
            color=choose_node_color(data),
            shape="dot",
            size=18,
            borderWidth=1,
        )

    for source, target, data in graph.edges(data=True):
        edge_type = infer_edge_type(data)
        net.add_edge(
            source=str(source),
            to=str(target),
            label=edge_type,
            title=f"<b>relation:</b> {edge_type}",
            color="#8A8A8A",
            arrows="to" if is_directed else "",
            smooth={"enabled": True, "type": "dynamic"},
            font={
                "size": 10,
                "align": "top",
                "vadjust": -8,
                "background": "rgba(255,255,255,0.85)",
                "strokeWidth": 0,
            },
        )

    net.set_options(
        """
        var options = {
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true
          },
          "nodes": {
            "font": {
              "size": 16,
              "face": "Arial"
            },
            "margin": 8
          },
          "edges": {
            "font": {
              "size": 13,
              "align": "top",
              "vadjust": -8,
              "background": "rgba(255,255,255,0.85)"
            },
            "color": {
              "inherit": false
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -120,
              "springLength": 220,
              "springConstant": 0.04,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "stabilization": {
              "enabled": true,
              "iterations": 1800
            }
          }
        }
        """
    )
    return net


def render_graph_html() -> str:
    gml_path = Path(GML_FILE_PATH).expanduser().resolve()
    if not gml_path.exists():
        raise FileNotFoundError(f"GML file not found: {gml_path}")

    net = build_network(gml_path)
    html = net.generate_html(notebook=False)
    freeze_script = """
<script type="text/javascript">
  // Stabilize first, then turn physics off so manual dragging stays in place.
  network.once("stabilizationIterationsDone", function () {
    network.setOptions({ physics: { enabled: false } });
  });
</script>
"""
    return html.replace("</body>", freeze_script + "\n</body>")


class GraphHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_response(404)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        try:
            html = render_graph_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Content-length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        except Exception as exc:
            err = f"Failed to render graph: {exc}".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.send_header("Content-length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)


def main() -> None:
    print(f"Serving graph for: {Path(GML_FILE_PATH).expanduser().resolve()}")
    print(f"Open: http://127.0.0.1:{PORT}")
    server = HTTPServer((HOST, PORT), GraphHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
