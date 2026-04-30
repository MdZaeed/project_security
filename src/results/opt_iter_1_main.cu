(int));
151:
152:   int *d_accuracy;
153:   cudaMalloc((void**)&d_accuracy, sizeof(int));
154:
155:   int *d_count_ref;
156:   cudaMalloc((void**)&d_count_ref, sizeof(int));
157:
158:   int *d_accuracy_ref;
159:   cudaMalloc((void**)&d_accuracy_ref, sizeof(int));
160:
161:   int *d_count_ref_2;
162:   cudaMalloc((void**)&d_count_ref_2, sizeof(int));
163:
164:   int *d_accuracy_ref_2;
165:   cudaMalloc((void**)&d_accuracy_ref_2, sizeof(int));
166:
167:   int *d_count_ref_3;
168:   cudaMalloc((void**)&d_count_ref_3, sizeof(int));
169:
170:   int *d_accuracy_ref_3;
171:   cudaMalloc((void**)&d_accuracy_ref_3, sizeof(int));
172:
173:   int *d_count_ref_4;
174:   cudaMalloc((void**)&d_count_ref_4, sizeof(int));
175:
176:   int *d_accuracy_ref_4;
177:   cudaMalloc((void**)&d_accuracy_ref_4, sizeof(int));
178:
179:   int *d_count_ref_5;
180:   cudaMalloc((void**)&d_count_ref_5, sizeof(int));
181:
182:   int *d_accuracy_ref_5;
183:   cudaMalloc((void