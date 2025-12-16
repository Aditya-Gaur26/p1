"""
Generate diagrams and figures from text descriptions
Supports Mermaid, PlantUML, ASCII art, and Graphviz
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
import base64
import subprocess
import tempfile

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class FigureGenerator:
    """Generate various types of diagrams and figures"""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize figure generator
        
        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = output_dir or (config.output_dir / "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check available tools
        self.mermaid_available = self._check_mermaid_cli()
        self.plantuml_available = self._check_plantuml()
        self.graphviz_available = self._check_graphviz()
    
    def _check_mermaid_cli(self) -> bool:
        """Check if Mermaid CLI is installed"""
        try:
            subprocess.run(['mmdc', '--version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def _check_plantuml(self) -> bool:
        """Check if PlantUML is available"""
        try:
            subprocess.run(['plantuml', '-version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def _check_graphviz(self) -> bool:
        """Check if Graphviz is installed"""
        try:
            subprocess.run(['dot', '-V'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def generate_mermaid_diagram(self, description: str, diagram_type: str = "flowchart") -> Dict[str, Any]:
        """
        Generate Mermaid diagram code from description
        
        Args:
            description: Text description of the diagram
            diagram_type: Type of diagram (flowchart, sequence, state, etc.)
            
        Returns:
            Dictionary with diagram code and metadata
        """
        # Use LLM to generate Mermaid code
        mermaid_code = self._generate_mermaid_code(description, diagram_type)
        
        # Try to render if CLI available
        image_path = None
        if self.mermaid_available:
            image_path = self._render_mermaid(mermaid_code)
        
        return {
            'type': 'mermaid',
            'diagram_type': diagram_type,
            'code': mermaid_code,
            'image_path': str(image_path) if image_path else None,
            'can_render': self.mermaid_available
        }
    
    def _generate_mermaid_code(self, description: str, diagram_type: str) -> str:
        """Generate Mermaid code based on description"""
        
        # Simple heuristic-based generation
        # In production, use LLM for better results
        
        if "tcp" in description.lower() and "handshake" in description.lower():
            return self._tcp_handshake_mermaid()
        
        elif "deadlock" in description.lower() or "circular wait" in description.lower():
            return self._deadlock_diagram_mermaid()
        
        elif "process state" in description.lower():
            return self._process_state_diagram_mermaid()
        
        elif "osi" in description.lower() and "layer" in description.lower():
            return self._osi_model_mermaid()
        
        else:
            # Generic flowchart
            return f"""flowchart TD
    A[Start: {description[:30]}]
    B[Process 1]
    C[Process 2]
    D[End]
    
    A --> B
    B --> C
    C --> D
    
    style A fill:#e1f5ff
    style D fill:#e1ffe1
"""
    
    def _tcp_handshake_mermaid(self) -> str:
        """Generate TCP handshake sequence diagram"""
        return """sequenceDiagram
    participant Client
    participant Server
    
    Note over Client,Server: TCP Three-Way Handshake
    
    Client->>Server: SYN (seq=x)
    Note right of Client: SYN_SENT state
    
    Server->>Client: SYN-ACK (seq=y, ack=x+1)
    Note left of Server: SYN_RECEIVED state
    
    Client->>Server: ACK (seq=x+1, ack=y+1)
    Note over Client,Server: Connection Established
    
    Note right of Client: ESTABLISHED state
    Note left of Server: ESTABLISHED state
"""
    
    def _deadlock_diagram_mermaid(self) -> str:
        """Generate deadlock circular wait diagram"""
        return """graph LR
    P1[Process P1] -->|Holds| R1[Resource R1]
    P1 -->|Waits for| R2[Resource R2]
    
    P2[Process P2] -->|Holds| R2
    P2 -->|Waits for| R3[Resource R3]
    
    P3[Process P3] -->|Holds| R3
    P3 -->|Waits for| R1
    
    style P1 fill:#ffcccc
    style P2 fill:#ffcccc
    style P3 fill:#ffcccc
    style R1 fill:#ccffcc
    style R2 fill:#ccffcc
    style R3 fill:#ccffcc
    
    classDef deadlock fill:#ffcccc,stroke:#ff0000,stroke-width:3px
    class P1,P2,P3 deadlock
"""
    
    def _process_state_diagram_mermaid(self) -> str:
        """Generate process state transition diagram"""
        return """stateDiagram-v2
    [*] --> New: Create Process
    New --> Ready: Admitted
    Ready --> Running: Scheduler Dispatch
    Running --> Ready: Interrupt
    Running --> Waiting: I/O or Event Wait
    Waiting --> Ready: I/O or Event Completion
    Running --> Terminated: Exit
    Terminated --> [*]
    
    note right of New
        Process created
        but not yet loaded
    end note
    
    note right of Running
        Process executing
        on CPU
    end note
"""
    
    def _osi_model_mermaid(self) -> str:
        """Generate OSI model diagram"""
        return """graph TD
    subgraph "OSI Model Layers"
        L7[7. Application Layer<br/>HTTP, FTP, SMTP]
        L6[6. Presentation Layer<br/>Encryption, Compression]
        L5[5. Session Layer<br/>Session Management]
        L4[4. Transport Layer<br/>TCP, UDP]
        L3[3. Network Layer<br/>IP, Routing]
        L2[2. Data Link Layer<br/>MAC, Switching]
        L1[1. Physical Layer<br/>Cables, Signals]
    end
    
    L7 --> L6
    L6 --> L5
    L5 --> L4
    L4 --> L3
    L3 --> L2
    L2 --> L1
    
    style L7 fill:#e3f2fd
    style L6 fill:#e8f5e9
    style L5 fill:#fff3e0
    style L4 fill:#fce4ec
    style L3 fill:#f3e5f5
    style L2 fill:#e0f2f1
    style L1 fill:#fafafa
"""
    
    def _render_mermaid(self, mermaid_code: str) -> Optional[Path]:
        """Render Mermaid code to image"""
        if not self.mermaid_available:
            return None
        
        try:
            # Create temp file with mermaid code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
                f.write(mermaid_code)
                temp_input = Path(f.name)
            
            # Output path
            output_path = self.output_dir / f"diagram_{hash(mermaid_code)}.png"
            
            # Render with mmdc
            subprocess.run([
                'mmdc',
                '-i', str(temp_input),
                '-o', str(output_path),
                '-b', 'transparent'
            ], check=True, capture_output=True)
            
            # Clean up temp file
            temp_input.unlink()
            
            return output_path
            
        except Exception as e:
            print(f"⚠️  Failed to render Mermaid: {e}")
            return None
    
    def generate_ascii_art(self, description: str) -> Dict[str, Any]:
        """
        Generate ASCII art diagram
        
        Args:
            description: Description of what to draw
            
        Returns:
            Dictionary with ASCII art
        """
        ascii_art = self._generate_ascii_diagram(description)
        
        return {
            'type': 'ascii',
            'art': ascii_art,
            'description': description
        }
    
    def _generate_ascii_diagram(self, description: str) -> str:
        """Generate ASCII art based on description"""
        
        if "tcp" in description.lower() and "handshake" in description.lower():
            return """
TCP Three-Way Handshake:

    Client                    Server
       |                         |
       |    SYN (seq=100)        |
       |------------------------>|
       |                         |
       |  SYN-ACK (seq=300,      |
       |          ack=101)       |
       |<------------------------|
       |                         |
       |    ACK (seq=101,        |
       |         ack=301)        |
       |------------------------>|
       |                         |
       |   [Connection Ready]    |
       |                         |
"""
        
        elif "deadlock" in description.lower():
            return """
Deadlock - Circular Wait:

    Process P1         Resource R1
        |                  |
        | <--- Holds ----- |
        |                  |
        | --- Waits --> Resource R2
                           |
                           | <--- Holds --- Process P2
                           |                    |
                      Resource R3 <--- Waits ---|
                           |
                           | <--- Holds --- Process P3
                           |                    |
    Resource R1 <--------- Waits ---------------|

[Circular dependency detected!]
"""
        
        elif "process" in description.lower() and "state" in description.lower():
            return """
Process State Diagram:

         +-------+
         |  NEW  |
         +-------+
             |
             | admitted
             v
         +-------+     interrupt    +----------+
    +--->| READY |<-----------------|  RUNNING |
    |    +-------+                  +----------+
    |        |                           |    |
    |        | dispatch                  |    | exit
    |        +-------------------------->|    |
    |                                    |    v
    |    +----------+    I/O complete    | +------------+
    +----| WAITING  |<-------------------+ | TERMINATED |
         +----------+                      +------------+
"""
        
        elif "osi" in description.lower():
            return """
OSI Model - 7 Layers:

    +---------------------------+
    |   7. Application Layer    |  (HTTP, FTP, SMTP)
    +---------------------------+
    |  6. Presentation Layer    |  (Encryption, Compression)
    +---------------------------+
    |    5. Session Layer       |  (Session Management)
    +---------------------------+
    |   4. Transport Layer      |  (TCP, UDP, Port Numbers)
    +---------------------------+
    |    3. Network Layer       |  (IP, Routing, Subnetting)
    +---------------------------+
    |   2. Data Link Layer      |  (MAC, Switching, Frames)
    +---------------------------+
    |   1. Physical Layer       |  (Cables, Signals, Bits)
    +---------------------------+
"""
        
        else:
            # Generic diagram
            return f"""
Generic Diagram: {description[:50]}

    +----------+
    |  Start   |
    +----------+
         |
         v
    +----------+
    | Process  |
    +----------+
         |
         v
    +----------+
    |   End    |
    +----------+
"""
    
    def generate_plantuml_diagram(self, description: str) -> Dict[str, Any]:
        """Generate PlantUML diagram"""
        
        plantuml_code = self._generate_plantuml_code(description)
        
        # Try to render if PlantUML available
        image_path = None
        if self.plantuml_available:
            image_path = self._render_plantuml(plantuml_code)
        
        return {
            'type': 'plantuml',
            'code': plantuml_code,
            'image_path': str(image_path) if image_path else None,
            'can_render': self.plantuml_available
        }
    
    def _generate_plantuml_code(self, description: str) -> str:
        """Generate PlantUML code"""
        
        if "sequence" in description.lower() or "tcp" in description.lower():
            return """@startuml
participant Client
participant Server

Client -> Server: SYN
Server -> Client: SYN-ACK
Client -> Server: ACK

note over Client, Server: Connection Established
@enduml"""
        
        else:
            return """@startuml
start
:Process Input;
:Execute Logic;
:Generate Output;
stop
@enduml"""
    
    def _render_plantuml(self, plantuml_code: str) -> Optional[Path]:
        """Render PlantUML to image"""
        if not self.plantuml_available:
            return None
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.puml', delete=False) as f:
                f.write(plantuml_code)
                temp_input = Path(f.name)
            
            # Render
            output_path = self.output_dir / f"plantuml_{hash(plantuml_code)}.png"
            subprocess.run([
                'plantuml',
                '-o', str(self.output_dir),
                str(temp_input)
            ], check=True, capture_output=True)
            
            temp_input.unlink()
            return output_path
            
        except Exception as e:
            print(f"⚠️  Failed to render PlantUML: {e}")
            return None
    
    def detect_diagram_need(self, question: str, answer: str) -> bool:
        """Detect if question/answer would benefit from a diagram"""
        
        diagram_keywords = [
            'draw', 'diagram', 'illustrate', 'show', 'visualize',
            'flowchart', 'graph', 'chart', 'figure', 'picture',
            'architecture', 'structure', 'topology', 'layout'
        ]
        
        question_lower = question.lower()
        
        return any(keyword in question_lower for keyword in diagram_keywords)
    
    def suggest_diagram_type(self, text: str) -> str:
        """Suggest appropriate diagram type based on text"""
        
        text_lower = text.lower()
        
        if 'sequence' in text_lower or 'handshake' in text_lower or 'protocol' in text_lower:
            return 'sequence'
        elif 'state' in text_lower or 'transition' in text_lower:
            return 'state'
        elif 'flow' in text_lower or 'algorithm' in text_lower or 'process' in text_lower:
            return 'flowchart'
        elif 'architecture' in text_lower or 'component' in text_lower:
            return 'component'
        elif 'network' in text_lower or 'topology' in text_lower:
            return 'network'
        elif 'class' in text_lower or 'object' in text_lower:
            return 'class'
        else:
            return 'flowchart'


def generate_diagram_for_answer(question: str, answer: str) -> Optional[Dict[str, Any]]:
    """
    Generate appropriate diagram for question/answer pair
    
    Args:
        question: User question
        answer: Model answer
        
    Returns:
        Dictionary with diagram information or None
    """
    generator = FigureGenerator()
    
    # Check if diagram is needed
    if not generator.detect_diagram_need(question, answer):
        return None
    
    # Determine diagram type
    diagram_type = generator.suggest_diagram_type(question + " " + answer)
    
    # Generate diagrams
    diagrams = []
    
    # Always generate ASCII (no dependencies)
    ascii_diagram = generator.generate_ascii_art(question)
    diagrams.append(ascii_diagram)
    
    # Generate Mermaid if available
    mermaid_diagram = generator.generate_mermaid_diagram(question, diagram_type)
    diagrams.append(mermaid_diagram)
    
    return {
        'has_diagram': True,
        'diagrams': diagrams,
        'diagram_type': diagram_type,
        'question': question
    }


def main():
    """Test figure generation"""
    generator = FigureGenerator()
    
    print("🎨 Figure Generator Test\n")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        "Explain the TCP three-way handshake",
        "Draw a process state diagram",
        "Show the deadlock circular wait condition",
        "Illustrate the OSI model layers"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case}")
        print("-" * 60)
        
        # ASCII art
        ascii_result = generator.generate_ascii_art(test_case)
        print("ASCII Diagram:")
        print(ascii_result['art'])
        
        # Mermaid
        mermaid_result = generator.generate_mermaid_diagram(test_case)
        print("\nMermaid Code (first 200 chars):")
        print(mermaid_result['code'][:200] + "...")
        print(f"Can render: {mermaid_result['can_render']}")
        if mermaid_result['image_path']:
            print(f"Saved to: {mermaid_result['image_path']}")


if __name__ == "__main__":
    main()
