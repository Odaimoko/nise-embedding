

## Structure of MPII-Video-Pose

[Link](http://pose.mpi-inf.mpg.de/art-track/).

- Folder: `images`.
  - Images named in the form of `sequence_{videoID}_{frameID}`, all placed in this folder.
- Annotation `mat` files, named `sequence_{videoID}`. Within each, it has three fields, `image`, `annorect`, and `imgnum`.
  - `image`: $1\times 1$ struct with image name in it.
  - `annorect` has following fields
    - `x1,y1,x2,y2`: the bounding box of a person's head
    - `score`: all $-1​$ since this is not prediction.
    - `scale`: Some don't have this information and some are $1.0$, so I ignore this.
    - `track_id`, the id of the person.
    - `annopoints` has field `point` which has 4 fields
      - `id`: joint type
      - `x,y`: coordinate.
      - `is_visible`, $1$ if the joint is visible else $0$.
  - `imgnum`: the index in the `images` folder.