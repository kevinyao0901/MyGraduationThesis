slam;
list = get_operable_objs;
if obj_name in list {
  approach obj_name;
  grasp obj_name;
  pos = get_obj_position "bowl";
  add_variable pos 2 0.1;
  set_end pos;
  release;
}
else {
  say "I cannot find the object specified.";
}
