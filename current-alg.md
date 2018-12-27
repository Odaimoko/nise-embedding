# ALG in FrameItem

## detect_human

```python
if use ground truth detection (config)
	use it
elif provided with prediction box
	use it
else
	detect boxes
if filter people when detect
	filter with HUMAN_THRES
```

## joint_prop

```python
if this is the first frame
	skip
##  preprocess

if use all gt joints to propagate and there are gt joints in the prev frame
	joints from previous frame, visibility of joints is used
    prev_box_scores is set to all ones
	joints scores is assigned to visibility
    
elif prev frame has joints
	if filter people when prop
		all four variables are generate from prev.unified_bboxes, filtered with PROP_HUMAN_THRES
	else
    	use all unified_bboxes
    prev_joints_vis = all ones
    
    if use gt joints to prop and prev frame has joints
        new joints are chosen as those 
        compute iou between prev frame (filtered) unified box and prev gt boxes
        match prev boxes and gt boxes, (prev unified box, prev gt box), similarity is IoU
		the pose of matched gt is used, scores are assigned as correpsonding prev unidifed boxes
# preprocess end
```

At this time, we have four variables

- `prev_frame_joints`: joints used to propagate, `num_people x num_joints x 2`
- `prev_frame_joints_vis`: visiblity of each joint, `num_people x num_joints`.
- `prop_joints_scores`: scores of each joint, used to assist proped boxes' scores
- `prop_boxes_scores`: scores to be assigned to proped boxes

Then generate scores

```python
for each joint set in prev_frame_joints:
    for each joint in prev_frame_joints:
        if old joint is inside the image and the visibility of old joint is True, then the corresponding new_joint_vis is True. Note that new_joint_vis is not the visibility of the newly proped joint
        if new_joint_vis is True
        	joint_flow = flow on the joint position
        else
        	joint_flow = 0
		new_proped_joint = joint_pos + joint_flow

```

Use all `new_proped_joints` to generate boxes. Boxes scores are determined as follows

-  mean of `prop_joints_scores` is calculated, along the people dimension
- `final_prop_box_scores` =  mean of `prop_joints_scores` * `prop_boxes_scores`

Assign `final_prop_box_scores` to proped boxes



## unify boxes(NMS)

```
if this is the first frame or task 1 is running:
	all_boxes = all detected boxes
else
	all boxes = detected and proped

if all boxes has nothing
	unified_box = empty
else
	NMS
```

NMS's procedure:

- take nms thres 1 = 0.05, thres 2 = 0.5

- filter those < thres 1

    - if nothing left, set empty boxes

- compute IoU between each pair

- sort the rest boxes scores in descending order.

- Let `KEEP` = empty list

    ```
    while we still have boxes
    	take the box with highest score
    	KEEP.append(highest)
    	for the rest boxes, keep those which has IoU <= thres 2 with highest
    ```

- unified boxes = KEEP





## get filtered boxes

```
if there is no unified box
	return empty tensor, None
else
	filter unifed boxes with thres _HUMAN_THRES
```



## est_joints

```python
if not (box is nmsed) or (this is the first frame) 
	return
for each box in unified box
	box score is obtained
    compute box center and scale using `box2cs`, set rotation=0
    obtain a transform matrix by center, scale, rotation and image size, image size is the size to input to the joint estimation model
    use cv2.warpAffine to get a cropped people
    detect joints, get the prediction's position and the corresponding heatmap value
	set new joints to the prediction position
    set new joints' score to be the heatmap values * box score
    
```

## assign_id

```
if ASSGIN_ID_TO_FILTERED_BOX
	filter this frame's unified boxes to be id_boxes
	id_idx_in_unified is set to the corresponding indices 
else
	id_boxes = all unified boxes
	id_idx set to 0~len(unified boxes)

people_ids = vector of zeros, size is id_boxes' length

if id_boxes is not empty

    if this is the first frame
        people_ids = range 1 to id_boxes' length	(inclusion)
        set static variable max_id = id_boxes' length.   max_id serves to assign to a new instance id
    else
        if prev frame has id
            proped_joints = this frame's new_joints
            proped_ids = previous frame's people id
	
```

From here we can see a problem: what if the number of proped joints is not the same with previous frame's id?

For now we use box iou, so no need to concern about joint prop.

```
            if matching metric is box iou
            dist_mat = tf_iou(self.id_boxes, prev_frame.idboxes)
            else:
            ...

            if matching alg is Munkres:
            use mkrs to match
            elif matching alg is greedy
            use bipartite_matching_greedy (from Detect-and-Track)

            for i_in_current, i_in_new_joints:
            people_ids[i_in_current] = prev_frame.people_ids[i_in_prev]
            for the rest in people_ids (there are unassigned instances)
            each is assigned a new id (max_id + 1) 

```



## to_dict

Output for evaluation

```
if task = 1 or 2
  {
    'image': [
      {
      'name': self.img_path,
      }
    ],
    'annorect': [  # i for people
      {
      'x1': [0],
      'x2': [0],
      'y1': [0],
      'y2': [0],
      'score': [-1],
      'track_id': [0],
      'annopoints': [
        {
        'point': [
          {  # j for joints
          'id': [j],
          'x': [self.joints[i, j, 0].item() * self.ori_img_w / self.img_w],
          'y': [self.joints[i, j, 1].item() * self.ori_img_h / self.img_h],
          'score': [self.joints_score[i, j].item()]
          } for j in range(num_joints)
        ]
        }
      ]
      } for i in range(unified_bboxes.shape[0])
    ]
  }
```

```
if task = 3
{
    'image': [
      {
      'name': self.img_path,
      }
    ],
    'annorect': [  # i for people
      {
      'x1': [0],
      'x2': [0],
      'y1': [0],
      'y2': [0],
      'score': [self.id_boxes[i, 4].item()],
      'track_id': [self.people_ids[i].item()],
      'annopoints': [
        {
        'point': [
          {  # j for joints
          'id': [j],
          'x': [self.joints[i, j, 0].item() * self.ori_img_w / self.img_w],
          'y': [self.joints[i, j, 1].item() * self.ori_img_h / self.img_h],
          'score': [self.joints_score[i, j].item()]
          } for j in range(nise_cfg.DATA.num_joints)
        ]
        }
      ]
      } for i in range(self.people_ids.shape[0])
    
    ]
    # 'imgnum'
}
```

