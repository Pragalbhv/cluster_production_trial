# Production Voronoi-Based Pu-Cl Cluster Analysis Specification

## Project Overview

This production code implements a comprehensive analysis pipeline for Pu-Cl cluster systems using Voronoi tessellation from OVITO. The system performs weighted coordination analysis, shared-anion network construction, oligomer characterization, and temporal stability analysis. We shpould be able to specify cation = 'Ce'/'Pu'. For now we will assume that we have specified cation as 'Pu'

## Core Objectives

1. **Voronoi Tessellation**: Use OVITO's Voronoi modifier to perform Voronoi tessellations. You will add VoronoiAnalysisModifier, with generate_bonds= True. You may want to compute surface mesh to get face areas 
  Possible references: voronoi/ovito/vornoi_ovito_cannon.ipynb 

2. **Weighted Coordination**: Use weights of the faces (area, solid angle) to do weighted coordination. Here we will build coordination histograms for Pu with (Cl, Na, Pu and Any)
  Possible reference: voronoi/frued/voronoi_canon.ipynb

3. **Build Shared Anion Graph**: Threshold minimum area as a percent of total area, treating connected faces as joined based on Voronoi tessellation faces. Nodes are Pu & Cl. Build Pu-Cl based network where Pu nodes define oligomer length
  Possible references: voronoi/ovito/vornoi_ovito_cannon.ipynb, pu_cluster_analysis_unified.ipynb

6. **Sharing Classification**: Determine if Pu-Pu connections via Cl atoms are face, edge, or corner sharing (3+, 2, or 1 Cl between them)

7. **Temporal Analysis**: Compute time-based quantities like cage correlation and oligomer stability

## Project Structure

```
cluster_production_trial/
├── spec.md                    # This specification document
├── utils.py                   # utility functions - functions that are used in notebook and are implemented/inspired in cluster_analysis
├── plots.py                  #has plotting utilities
├── cluster_analysis.ipynb  # Interactive analysis notebook
├── filesystem_changes.log     # Log of all file system changes
└── .gitignore                # Git ignore patterns
```


## File System Change Logging

### Requirement
**Every change in the file system within `cluster_production_trial` directory must be logged.**

#### Logging System
- **Log File**: `filesystem_changes.log` in project root
- **Format**: 
  ```
  [TIMESTAMP] [ACTION] [FILE_PATH] [DETAILS]
  ```
- **Actions to Log**:
  - File creation
  - File modification
  - File deletion
  - Directory creation
  - Directory deletion
  - File moves/renames

#### Logging Decorator/Context Manager
- Create utility decorator for functions that modify files
- Create context manager for file operations
- Automatically log all file system changes

#### Log Structure Example
```
[2024-01-15 10:30:45] CREATE production_utils.py [size: 15234 bytes]
[2024-01-15 10:31:12] MODIFY production_utils.py [lines changed: 45-67]
[2024-01-15 10:32:00] CREATE production_analysis.ipynb [size: 89321 bytes]
[2024-01-15 10:33:15] DELETE temp_data.pkl [reason: cleanup]
[2024-01-15 10:34:20] CREATE subdirectory/ [directory]
```

## Version Control

### Git Workflow
- All code changes tracked via git
- Regular commits with descriptive messages
- Feature branches for major additions
- Tagged releases for major milestones

### File System Logging
- All file system changes logged via `filesystem_changes.log`
- Log file itself should be tracked in git
- Regular review of log file for project activity

### Change Tracking
- Code changes: Git commits
- File system changes: filesystem_changes.log
- Analysis results: Timestamped output files
- Notebooks: Versioned with cell outputs

## Development Workflow

### Phase 1: Core Infrastructure
1. Set up project structure
2. Implement file logging system (`filesystem_changes.log`)
3. Basic OVITO pipeline setup and data loading utilities
4. Test with simple trajectory files

### Phase 2: Voronoi Tessellation
1. Implement `VoronoiAnalysisModifier` application with `generate_bonds=True`
2. Extract `DataCollection` output from OVITO pipeline
3. Access Bonds object with topology (Voronoi face connections)
4. Optionally compute surface mesh to extract face areas
5. Extract particle properties: Atomic Volume, Coordination, Voronoi Index
6. Reference: `voronoi/ovito/vornoi_ovito_cannon.ipynb`
7. Test tessellation with sample data

### Phase 3: Weighted Coordination Analysis
1. Extract face areas from Voronoi bonds or surface mesh
2. Compute solid angles (Ω = A / r²) for each face
3. Implement area-weighted coordination (CN_A)
4. Implement solid-angle-weighted coordination (CN_Ω)
5. Build coordination histograms for Pu with neighbors:
   - Pu-Cl coordination histogram
   - Pu-Na coordination histogram
   - Pu-Pu coordination histogram
   - Pu-Any (total) coordination histogram
6. Reference: `voronoi/frued/voronoi_canon.ipynb`
7. Validate coordination calculations

### Phase 4: Area Thresholding and Shared Anion Graph
1. Compute total Voronoi cell area per atom
2. Implement percent-based area thresholding (min_area_percent parameter)
3. Implement connected face merging logic (treat connected faces as joined)
4. Filter bonds/faces based on threshold
5. Build shared anion graph from Voronoi tessellation faces:
   - Nodes: Pu and Cl atoms
   - Edges: Connections via shared Voronoi faces
6. Construct Pu-Cl network where Pu nodes define oligomer length
7. References: `voronoi/ovito/vornoi_ovito_cannon.ipynb`, `pu_cluster_analysis_unified.ipynb`
8. Test graph construction and oligomer identification

### Phase 5: Sharing Classification
1. Implement Pu-Pu connection analysis via Cl atoms
2. Count number of Cl atoms connecting each Pu-Pu pair:
   - Face sharing: 3+ Cl atoms
   - Edge sharing: 2 Cl atoms
   - Corner sharing: 1 Cl atom
3. Classify all Pu-Pu pairs in network
4. Generate statistics by sharing type
5. Validate classification logic

### Phase 6: Temporal Analysis
1. Implement cage correlation calculation:
   - Compare coordination environments between frames
   - Compute correlation coefficients
2. Implement oligomer stability tracking:
   - Track oligomer persistence over time
   - Compute lifetimes, breakage rates, formation rates
   - Analyze size evolution
3. Test temporal tracking with multi-frame trajectories
4. Optimize for large trajectory files

### Phase 7: Notebook Development
1. Create `production_analysis.ipynb` structure
2. Implement sections:
   - Setup and configuration
   - Data loading
   - Voronoi tessellation
   - Weighted coordination histograms
   - Shared anion graph construction
   - Oligomer analysis
   - Sharing type classification
   - Temporal analysis
3. Add visualizations for each analysis step
4. Test end-to-end workflow
5. Document usage examples

### Phase 8: Testing and Documentation
1. Write unit tests for each utility function
2. Write integration tests for full pipeline
3. Validate against reference notebooks
4. Complete function docstrings
5. Performance optimization for large systems
6. Create usage documentation


**Last Updated**: [Date will be updated when spec is modified]


