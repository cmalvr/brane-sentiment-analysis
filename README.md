# brane-sentiment-analysis

To run this repository you need the updated brane files from our local branch. The following commands need to be run from the brane-sentiment-analysis main directory.

## Building data
```
brane data build data/train/data.yml  
brane data build data/test/data.yml
```

## Building packages
```
brane package build packages/compute/container.yml 
brane package build packages/display/container.yml
```

## Running the pipeline
```
brane workflow run pipeline.bs
```

Results will be fond at “/Users/user-name/Library/Application Support/brane/data” or wherever brane DFS is defined.
