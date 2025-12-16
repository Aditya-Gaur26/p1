"""
Concept mapper - identifies related concepts and prerequisites
"""

import sys
from pathlib import Path
from typing import List, Dict, Set

sys.path.append(str(Path(__file__).parent.parent.parent))


class ConceptMapper:
    """Map relationships between OS/Network concepts"""
    
    def __init__(self):
        # Define concept relationships
        self.concept_graph = {
            # Process Management
            'process': {
                'related': ['thread', 'program', 'context_switch', 'pcb'],
                'prerequisites': ['program', 'cpu'],
                'subtopics': ['process_state', 'process_creation', 'process_termination'],
                'category': 'process_management'
            },
            'thread': {
                'related': ['process', 'concurrency', 'multithreading'],
                'prerequisites': ['process'],
                'subtopics': ['user_thread', 'kernel_thread', 'thread_pool'],
                'category': 'process_management'
            },
            'scheduling': {
                'related': ['process', 'cpu', 'algorithm'],
                'prerequisites': ['process', 'cpu'],
                'subtopics': ['fcfs', 'sjf', 'round_robin', 'priority_scheduling'],
                'category': 'cpu_scheduling'
            },
            
            # Synchronization
            'synchronization': {
                'related': ['process', 'thread', 'critical_section'],
                'prerequisites': ['process', 'thread'],
                'subtopics': ['mutex', 'semaphore', 'monitor'],
                'category': 'synchronization'
            },
            'deadlock': {
                'related': ['synchronization', 'resource', 'process'],
                'prerequisites': ['process', 'resource_allocation'],
                'subtopics': ['deadlock_prevention', 'deadlock_avoidance', 'bankers_algorithm'],
                'category': 'synchronization'
            },
            'semaphore': {
                'related': ['synchronization', 'mutex'],
                'prerequisites': ['synchronization', 'critical_section'],
                'subtopics': ['binary_semaphore', 'counting_semaphore'],
                'category': 'synchronization'
            },
            
            # Memory Management
            'memory': {
                'related': ['ram', 'address', 'allocation'],
                'prerequisites': [],
                'subtopics': ['physical_memory', 'virtual_memory', 'memory_management'],
                'category': 'memory_management'
            },
            'virtual_memory': {
                'related': ['memory', 'paging', 'segmentation'],
                'prerequisites': ['memory', 'address'],
                'subtopics': ['page_table', 'tlb', 'page_replacement'],
                'category': 'memory_management'
            },
            'paging': {
                'related': ['virtual_memory', 'page_table'],
                'prerequisites': ['memory', 'address'],
                'subtopics': ['page_fault', 'page_replacement', 'demand_paging'],
                'category': 'memory_management'
            },
            
            # Networking
            'tcp': {
                'related': ['transport_layer', 'ip', 'protocol'],
                'prerequisites': ['network', 'protocol'],
                'subtopics': ['tcp_handshake', 'tcp_congestion', 'tcp_flow_control'],
                'category': 'networking'
            },
            'udp': {
                'related': ['transport_layer', 'tcp', 'protocol'],
                'prerequisites': ['network', 'protocol'],
                'subtopics': [],
                'category': 'networking'
            },
            'ip': {
                'related': ['network_layer', 'routing', 'address'],
                'prerequisites': ['network'],
                'subtopics': ['ipv4', 'ipv6', 'ip_routing'],
                'category': 'networking'
            },
            'osi_model': {
                'related': ['network', 'protocol', 'layer'],
                'prerequisites': ['network'],
                'subtopics': ['physical_layer', 'data_link', 'network_layer', 'transport_layer'],
                'category': 'networking'
            },
        }
    
    def get_related_concepts(self, concept: str) -> List[str]:
        """Get concepts related to the given concept"""
        concept_lower = concept.lower().replace(' ', '_')
        
        if concept_lower in self.concept_graph:
            return self.concept_graph[concept_lower].get('related', [])
        
        # Search for partial matches
        related = set()
        for key, value in self.concept_graph.items():
            if concept_lower in key or key in concept_lower:
                related.update(value.get('related', []))
        
        return list(related)
    
    def get_prerequisites(self, concept: str) -> List[str]:
        """Get prerequisite concepts"""
        concept_lower = concept.lower().replace(' ', '_')
        
        if concept_lower in self.concept_graph:
            return self.concept_graph[concept_lower].get('prerequisites', [])
        
        # Search for partial matches
        prereqs = set()
        for key, value in self.concept_graph.items():
            if concept_lower in key or key in concept_lower:
                prereqs.update(value.get('prerequisites', []))
        
        return list(prereqs)
    
    def get_subtopics(self, concept: str) -> List[str]:
        """Get subtopics of a concept"""
        concept_lower = concept.lower().replace(' ', '_')
        
        if concept_lower in self.concept_graph:
            return self.concept_graph[concept_lower].get('subtopics', [])
        
        return []
    
    def get_learning_path(self, target_concept: str) -> List[str]:
        """Get recommended learning path to reach target concept"""
        concept_lower = target_concept.lower().replace(' ', '_')
        
        if concept_lower not in self.concept_graph:
            return [target_concept]
        
        # Simple BFS to find prerequisites
        path = []
        visited = set()
        queue = [(concept_lower, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            prereqs = self.concept_graph.get(current, {}).get('prerequisites', [])
            
            for prereq in prereqs:
                if prereq not in visited:
                    queue.append((prereq, depth + 1))
            
            path.append((current, depth))
        
        # Sort by depth (prerequisites first)
        path.sort(key=lambda x: x[1])
        
        return [p[0].replace('_', ' ').title() for p, _ in path]
    
    def get_concept_info(self, concept: str) -> Dict:
        """Get comprehensive information about a concept"""
        concept_lower = concept.lower().replace(' ', '_')
        
        return {
            'concept': concept,
            'related': self.get_related_concepts(concept),
            'prerequisites': self.get_prerequisites(concept),
            'subtopics': self.get_subtopics(concept),
            'learning_path': self.get_learning_path(concept),
            'category': self.concept_graph.get(concept_lower, {}).get('category', 'unknown')
        }


def main():
    """Test concept mapper"""
    print("=" * 60)
    print("Testing Concept Mapper".center(60))
    print("=" * 60)
    
    mapper = ConceptMapper()
    
    test_concepts = ['deadlock', 'virtual memory', 'tcp']
    
    for concept in test_concepts:
        print(f"\n📚 Concept: {concept.upper()}")
        info = mapper.get_concept_info(concept)
        
        print(f"\nCategory: {info['category']}")
        
        if info['prerequisites']:
            print(f"\nPrerequisites: {', '.join(info['prerequisites'])}")
        
        if info['related']:
            print(f"Related: {', '.join(info['related'])}")
        
        if info['subtopics']:
            print(f"Subtopics: {', '.join(info['subtopics'])}")
        
        print(f"\nLearning Path: {' → '.join(info['learning_path'])}")


if __name__ == "__main__":
    main()
