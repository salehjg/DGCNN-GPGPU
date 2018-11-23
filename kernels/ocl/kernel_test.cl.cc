kernel void vectorAddition(global read_only int* vector1, global read_only int* vector2, global write_only int* vector3)
{
    int indx = get_global_id(0);
    vector3[indx] = vector1[indx] + vector2[indx];
}