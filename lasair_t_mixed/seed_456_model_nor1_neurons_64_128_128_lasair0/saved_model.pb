Ѥ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02unknown8��
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0
�
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_c2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_c2/bias/v
y
(Adam/dense_c2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_c2/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_c2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/dense_c2/kernel/v
�
*Adam/dense_c2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_c2/kernel/v*
_output_shapes
:	� *
dtype0
�
Adam/dense_c1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_c1/bias/v
z
(Adam/dense_c1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_c1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_c1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_c1/kernel/v
�
*Adam/dense_c1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_c1/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_me2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/dense_me2/bias/v
|
)Adam/dense_me2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_me2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_me2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/dense_me2/kernel/v
�
+Adam/dense_me2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_me2/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_me1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/dense_me1/bias/v
|
)Adam/dense_me1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_me1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_me1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/dense_me1/kernel/v
�
+Adam/dense_me1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_me1/kernel/v*
_output_shapes
:	�*
dtype0
}
Adam/conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/conv_2/bias/v
v
&Adam/conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_nameAdam/conv_2/kernel/v
�
(Adam/conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/v*(
_output_shapes
:��*
dtype0
}
Adam/conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/conv_1/bias/v
v
&Adam/conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*%
shared_nameAdam/conv_1/kernel/v
�
(Adam/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/v*'
_output_shapes
:@�*
dtype0
|
Adam/conv_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_0/bias/v
u
&Adam/conv_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_0/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv_0/kernel/v
�
(Adam/conv_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_0/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
�
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_c2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_c2/bias/m
y
(Adam/dense_c2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_c2/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_c2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/dense_c2/kernel/m
�
*Adam/dense_c2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_c2/kernel/m*
_output_shapes
:	� *
dtype0
�
Adam/dense_c1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_c1/bias/m
z
(Adam/dense_c1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_c1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_c1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_c1/kernel/m
�
*Adam/dense_c1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_c1/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_me2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/dense_me2/bias/m
|
)Adam/dense_me2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_me2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_me2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/dense_me2/kernel/m
�
+Adam/dense_me2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_me2/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_me1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/dense_me1/bias/m
|
)Adam/dense_me1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_me1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_me1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/dense_me1/kernel/m
�
+Adam/dense_me1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_me1/kernel/m*
_output_shapes
:	�*
dtype0
}
Adam/conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/conv_2/bias/m
v
&Adam/conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_nameAdam/conv_2/kernel/m
�
(Adam/conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/m*(
_output_shapes
:��*
dtype0
}
Adam/conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/conv_1/bias/m
v
&Adam/conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*%
shared_nameAdam/conv_1/kernel/m
�
(Adam/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/m*'
_output_shapes
:@�*
dtype0
|
Adam/conv_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_0/bias/m
u
&Adam/conv_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_0/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv_0/kernel/m
�
(Adam/conv_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_0/kernel/m*&
_output_shapes
:@*
dtype0
�
*data_augmentation/random_rotation/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*;
shared_name,*data_augmentation/random_rotation/StateVar
�
>data_augmentation/random_rotation/StateVar/Read/ReadVariableOpReadVariableOp*data_augmentation/random_rotation/StateVar*
_output_shapes
:*
dtype0	
�
&data_augmentation/random_flip/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*7
shared_name(&data_augmentation/random_flip/StateVar
�
:data_augmentation/random_flip/StateVar/Read/ReadVariableOpReadVariableOp&data_augmentation/random_flip/StateVar*
_output_shapes
:*
dtype0	
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

: *
dtype0
r
dense_c2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_c2/bias
k
!dense_c2/bias/Read/ReadVariableOpReadVariableOpdense_c2/bias*
_output_shapes
: *
dtype0
{
dense_c2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� * 
shared_namedense_c2/kernel
t
#dense_c2/kernel/Read/ReadVariableOpReadVariableOpdense_c2/kernel*
_output_shapes
:	� *
dtype0
s
dense_c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_c1/bias
l
!dense_c1/bias/Read/ReadVariableOpReadVariableOpdense_c1/bias*
_output_shapes	
:�*
dtype0
|
dense_c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_c1/kernel
u
#dense_c1/kernel/Read/ReadVariableOpReadVariableOpdense_c1/kernel* 
_output_shapes
:
��*
dtype0
u
dense_me2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_me2/bias
n
"dense_me2/bias/Read/ReadVariableOpReadVariableOpdense_me2/bias*
_output_shapes	
:�*
dtype0
~
dense_me2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namedense_me2/kernel
w
$dense_me2/kernel/Read/ReadVariableOpReadVariableOpdense_me2/kernel* 
_output_shapes
:
��*
dtype0
u
dense_me1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_me1/bias
n
"dense_me1/bias/Read/ReadVariableOpReadVariableOpdense_me1/bias*
_output_shapes	
:�*
dtype0
}
dense_me1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namedense_me1/kernel
v
$dense_me1/kernel/Read/ReadVariableOpReadVariableOpdense_me1/kernel*
_output_shapes
:	�*
dtype0
�
feature_ranking/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namefeature_ranking/kernel
}
*feature_ranking/kernel/Read/ReadVariableOpReadVariableOpfeature_ranking/kernel*
_output_shapes
:*
dtype0
o
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv_2/bias
h
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes	
:�*
dtype0
�
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameconv_2/kernel
y
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*(
_output_shapes
:��*
dtype0
o
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv_1/bias
h
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes	
:�*
dtype0

conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_nameconv_1/kernel
x
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*'
_output_shapes
:@�*
dtype0
n
conv_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_0/bias
g
conv_0/bias/Read/ReadVariableOpReadVariableOpconv_0/bias*
_output_shapes
:@*
dtype0
~
conv_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_0/kernel
w
!conv_0/kernel/Read/ReadVariableOpReadVariableOpconv_0/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_image_inputPlaceholder*/
_output_shapes
:���������<<*
dtype0*$
shape:���������<<
}
serving_default_meta_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_image_inputserving_default_meta_inputconv_0/kernelconv_0/biasconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasdense_me1/kerneldense_me1/biasdense_me2/kerneldense_me2/biasdense_c1/kerneldense_c1/biasdense_c2/kerneldense_c2/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_616950

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ɗ
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
neurons
	
label_dict


cnn_layers
feature_importance
data_augmentation
flatten
meta_weighted
dense_m1
dense_m2
concatenate
dense_c1
dense_c2
output_layer
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16*
�
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16*
* 
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
-trace_0
.trace_1
/trace_2
0trace_3* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
* 

50
61
72* 
* 

80
91
:2*
* 
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
AResizing
B
RandomFlip
CRandomRotation*
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

kernel
bias*
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

 kernel
!bias*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

"kernel
#bias*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

$kernel
%bias*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

&kernel
'bias*
�
titer

ubeta_1

vbeta_2
	wdecaym�m�m�m�m�m�m�m� m�!m�"m�#m�$m�%m�&m�'m�v�v�v�v�v�v�v�v� v�!v�"v�#v�$v�%v�&v�'v�*

xserving_default* 
MG
VARIABLE_VALUEconv_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfeature_ranking/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_me1/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_me1/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_me2/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_me2/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_c1/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_c1/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_c2/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_c2/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEoutput/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEoutput/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
y0
z1
{2
|3
}4
~5
6
7
8
9
10
11
12
13
14*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

y0
z1*

{0
|1*

}0
~1*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�factor*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

 0
!1*

 0
!1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

$0
%1*

$0
%1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

&0
'1*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
<
�	variables
�	keras_api

�total

�count*
* 

A0
B1
C2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�
_generator*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�
_generator*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�
_state_var*
* 
* 
* 
* 
* 

�
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
��
VARIABLE_VALUE&data_augmentation/random_flip/StateVar_data_augmentation/RandomFlip/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*data_augmentation/random_rotation/StateVarcdata_augmentation/RandomRotation/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_me1/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_me1/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_me2/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_me2/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_c1/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_c1/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_c2/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_c2/bias/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/output/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/output/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_me1/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_me1/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_me2/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_me2/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_c1/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_c1/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_c2/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_c2/bias/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/output/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/output/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv_0/kernelconv_0/biasconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasfeature_ranking/kerneldense_me1/kerneldense_me1/biasdense_me2/kerneldense_me2/biasdense_c1/kerneldense_c1/biasdense_c2/kerneldense_c2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount&data_augmentation/random_flip/StateVar*data_augmentation/random_rotation/StateVarAdam/conv_0/kernel/mAdam/conv_0/bias/mAdam/conv_1/kernel/mAdam/conv_1/bias/mAdam/conv_2/kernel/mAdam/conv_2/bias/mAdam/dense_me1/kernel/mAdam/dense_me1/bias/mAdam/dense_me2/kernel/mAdam/dense_me2/bias/mAdam/dense_c1/kernel/mAdam/dense_c1/bias/mAdam/dense_c2/kernel/mAdam/dense_c2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv_0/kernel/vAdam/conv_0/bias/vAdam/conv_1/kernel/vAdam/conv_1/bias/vAdam/conv_2/kernel/vAdam/conv_2/bias/vAdam/dense_me1/kernel/vAdam/dense_me1/bias/vAdam/dense_me2/kernel/vAdam/dense_me2/bias/vAdam/dense_c1/kernel/vAdam/dense_c1/bias/vAdam/dense_c2/kernel/vAdam/dense_c2/bias/vAdam/output/kernel/vAdam/output/bias/vConst*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_618243
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_0/kernelconv_0/biasconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasfeature_ranking/kerneldense_me1/kerneldense_me1/biasdense_me2/kerneldense_me2/biasdense_c1/kerneldense_c1/biasdense_c2/kerneldense_c2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount&data_augmentation/random_flip/StateVar*data_augmentation/random_rotation/StateVarAdam/conv_0/kernel/mAdam/conv_0/bias/mAdam/conv_1/kernel/mAdam/conv_1/bias/mAdam/conv_2/kernel/mAdam/conv_2/bias/mAdam/dense_me1/kernel/mAdam/dense_me1/bias/mAdam/dense_me2/kernel/mAdam/dense_me2/bias/mAdam/dense_c1/kernel/mAdam/dense_c1/bias/mAdam/dense_c2/kernel/mAdam/dense_c2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv_0/kernel/vAdam/conv_0/bias/vAdam/conv_1/kernel/vAdam/conv_1/bias/vAdam/conv_2/kernel/vAdam/conv_2/bias/vAdam/dense_me1/kernel/vAdam/dense_me1/bias/vAdam/dense_me2/kernel/vAdam/dense_me2/bias/vAdam/dense_c1/kernel/vAdam/dense_c1/bias/vAdam/dense_c2/kernel/vAdam/dense_c2/bias/vAdam/output/kernel/vAdam/output/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_618424��
�A
�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616583

inputs
inputs_1&
data_augmentation_616531:	&
data_augmentation_616533:	'
conv_0_616536:@
conv_0_616538:@(
conv_1_616542:@�
conv_1_616544:	�)
conv_2_616548:��
conv_2_616550:	�#
dense_me1_616556:	�
dense_me1_616558:	�$
dense_me2_616561:
��
dense_me2_616563:	�#
dense_c1_616567:
��
dense_c1_616569:	�"
dense_c2_616572:	� 
dense_c2_616574: 
output_616577: 
output_616579:
identity��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�)data_augmentation/StatefulPartitionedCall� dense_c1/StatefulPartitionedCall� dense_c2/StatefulPartitionedCall�!dense_me1/StatefulPartitionedCall�!dense_me2/StatefulPartitionedCall�output/StatefulPartitionedCall�
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinputsdata_augmentation_616531data_augmentation_616533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616294�
conv_0/StatefulPartitionedCallStatefulPartitionedCall2data_augmentation/StatefulPartitionedCall:output:0conv_0_616536conv_0_616538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������::@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_616311�
pool_0/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_0_layer_call_and_return_conditional_losses_616032�
conv_1/StatefulPartitionedCallStatefulPartitionedCallpool_0/PartitionedCall:output:0conv_1_616542conv_1_616544*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_616329�
pool_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_616044�
conv_2/StatefulPartitionedCallStatefulPartitionedCallpool_1/PartitionedCall:output:0conv_2_616548conv_2_616550*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_616347�
pool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_616056�
flatten/PartitionedCallPartitionedCallpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_616360�
feature_ranking/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_feature_ranking_layer_call_and_return_conditional_losses_616368�
!dense_me1/StatefulPartitionedCallStatefulPartitionedCall(feature_ranking/PartitionedCall:output:0dense_me1_616556dense_me1_616558*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me1_layer_call_and_return_conditional_losses_616381�
!dense_me2/StatefulPartitionedCallStatefulPartitionedCall*dense_me1/StatefulPartitionedCall:output:0dense_me2_616561dense_me2_616563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me2_layer_call_and_return_conditional_losses_616398�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*dense_me2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_616411�
 dense_c1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_c1_616567dense_c1_616569*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c1_layer_call_and_return_conditional_losses_616424�
 dense_c2/StatefulPartitionedCallStatefulPartitionedCall)dense_c1/StatefulPartitionedCall:output:0dense_c2_616572dense_c2_616574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c2_layer_call_and_return_conditional_losses_616441�
output/StatefulPartitionedCallStatefulPartitionedCall)dense_c2/StatefulPartitionedCall:output:0output_616577output_616579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_616458v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall*^data_augmentation/StatefulPartitionedCall!^dense_c1/StatefulPartitionedCall!^dense_c2/StatefulPartitionedCall"^dense_me1/StatefulPartitionedCall"^dense_me2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������<<:���������: : : : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_c1/StatefulPartitionedCall dense_c1/StatefulPartitionedCall2D
 dense_c2/StatefulPartitionedCall dense_c2/StatefulPartitionedCall2F
!dense_me1/StatefulPartitionedCall!dense_me1/StatefulPartitionedCall2F
!dense_me2/StatefulPartitionedCall!dense_me2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
�
5__inference_transient_classifier_layer_call_fn_616992
inputs_image_input
inputs_meta_input
unknown:	
	unknown_0:	#
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	� 

unknown_14: 

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_image_inputinputs_meta_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������<<:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_meta_input:c _
/
_output_shapes
:���������<<
,
_user_specified_nameinputs_image_input
�?
�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616523
image_input

meta_input'
conv_0_616476:@
conv_0_616478:@(
conv_1_616482:@�
conv_1_616484:	�)
conv_2_616488:��
conv_2_616490:	�#
dense_me1_616496:	�
dense_me1_616498:	�$
dense_me2_616501:
��
dense_me2_616503:	�#
dense_c1_616507:
��
dense_c1_616509:	�"
dense_c2_616512:	� 
dense_c2_616514: 
output_616517: 
output_616519:
identity��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall� dense_c1/StatefulPartitionedCall� dense_c2/StatefulPartitionedCall�!dense_me1/StatefulPartitionedCall�!dense_me2/StatefulPartitionedCall�output/StatefulPartitionedCall�
!data_augmentation/PartitionedCallPartitionedCallimage_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616474�
conv_0/StatefulPartitionedCallStatefulPartitionedCall*data_augmentation/PartitionedCall:output:0conv_0_616476conv_0_616478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������::@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_616311�
pool_0/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_0_layer_call_and_return_conditional_losses_616032�
conv_1/StatefulPartitionedCallStatefulPartitionedCallpool_0/PartitionedCall:output:0conv_1_616482conv_1_616484*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_616329�
pool_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_616044�
conv_2/StatefulPartitionedCallStatefulPartitionedCallpool_1/PartitionedCall:output:0conv_2_616488conv_2_616490*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_616347�
pool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_616056�
flatten/PartitionedCallPartitionedCallpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_616360�
feature_ranking/PartitionedCallPartitionedCall
meta_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_feature_ranking_layer_call_and_return_conditional_losses_616368�
!dense_me1/StatefulPartitionedCallStatefulPartitionedCall(feature_ranking/PartitionedCall:output:0dense_me1_616496dense_me1_616498*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me1_layer_call_and_return_conditional_losses_616381�
!dense_me2/StatefulPartitionedCallStatefulPartitionedCall*dense_me1/StatefulPartitionedCall:output:0dense_me2_616501dense_me2_616503*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me2_layer_call_and_return_conditional_losses_616398�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*dense_me2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_616411�
 dense_c1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_c1_616507dense_c1_616509*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c1_layer_call_and_return_conditional_losses_616424�
 dense_c2/StatefulPartitionedCallStatefulPartitionedCall)dense_c1/StatefulPartitionedCall:output:0dense_c2_616512dense_c2_616514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c2_layer_call_and_return_conditional_losses_616441�
output/StatefulPartitionedCallStatefulPartitionedCall)dense_c2/StatefulPartitionedCall:output:0output_616517output_616519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_616458v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall!^dense_c1/StatefulPartitionedCall!^dense_c2/StatefulPartitionedCall"^dense_me1/StatefulPartitionedCall"^dense_me2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2D
 dense_c1/StatefulPartitionedCall dense_c1/StatefulPartitionedCall2D
 dense_c2/StatefulPartitionedCall dense_c2/StatefulPartitionedCall2F
!dense_me1/StatefulPartitionedCall!dense_me1/StatefulPartitionedCall2F
!dense_me2/StatefulPartitionedCall!dense_me2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:SO
'
_output_shapes
:���������
$
_user_specified_name
meta_input:\ X
/
_output_shapes
:���������<<
%
_user_specified_nameimage_input
�
�
5__inference_transient_classifier_layer_call_fn_616712
image_input

meta_input!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	� 

unknown_12: 

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimage_input
meta_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:SO
'
_output_shapes
:���������
$
_user_specified_name
meta_input:\ X
/
_output_shapes
:���������<<
%
_user_specified_nameimage_input
�
q
G__inference_concatenate_layer_call_and_return_conditional_losses_616411

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616677

inputs
inputs_1'
conv_0_616630:@
conv_0_616632:@(
conv_1_616636:@�
conv_1_616638:	�)
conv_2_616642:��
conv_2_616644:	�#
dense_me1_616650:	�
dense_me1_616652:	�$
dense_me2_616655:
��
dense_me2_616657:	�#
dense_c1_616661:
��
dense_c1_616663:	�"
dense_c2_616666:	� 
dense_c2_616668: 
output_616671: 
output_616673:
identity��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall� dense_c1/StatefulPartitionedCall� dense_c2/StatefulPartitionedCall�!dense_me1/StatefulPartitionedCall�!dense_me2/StatefulPartitionedCall�output/StatefulPartitionedCall�
!data_augmentation/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616474�
conv_0/StatefulPartitionedCallStatefulPartitionedCall*data_augmentation/PartitionedCall:output:0conv_0_616630conv_0_616632*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������::@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_616311�
pool_0/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_0_layer_call_and_return_conditional_losses_616032�
conv_1/StatefulPartitionedCallStatefulPartitionedCallpool_0/PartitionedCall:output:0conv_1_616636conv_1_616638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_616329�
pool_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_616044�
conv_2/StatefulPartitionedCallStatefulPartitionedCallpool_1/PartitionedCall:output:0conv_2_616642conv_2_616644*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_616347�
pool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_616056�
flatten/PartitionedCallPartitionedCallpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_616360�
feature_ranking/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_feature_ranking_layer_call_and_return_conditional_losses_616368�
!dense_me1/StatefulPartitionedCallStatefulPartitionedCall(feature_ranking/PartitionedCall:output:0dense_me1_616650dense_me1_616652*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me1_layer_call_and_return_conditional_losses_616381�
!dense_me2/StatefulPartitionedCallStatefulPartitionedCall*dense_me1/StatefulPartitionedCall:output:0dense_me2_616655dense_me2_616657*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me2_layer_call_and_return_conditional_losses_616398�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*dense_me2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_616411�
 dense_c1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_c1_616661dense_c1_616663*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c1_layer_call_and_return_conditional_losses_616424�
 dense_c2/StatefulPartitionedCallStatefulPartitionedCall)dense_c1/StatefulPartitionedCall:output:0dense_c2_616666dense_c2_616668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c2_layer_call_and_return_conditional_losses_616441�
output/StatefulPartitionedCallStatefulPartitionedCall)dense_c2/StatefulPartitionedCall:output:0output_616671output_616673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_616458v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall!^dense_c1/StatefulPartitionedCall!^dense_c2/StatefulPartitionedCall"^dense_me1/StatefulPartitionedCall"^dense_me2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2D
 dense_c1/StatefulPartitionedCall dense_c1/StatefulPartitionedCall2D
 dense_c2/StatefulPartitionedCall dense_c2/StatefulPartitionedCall2F
!dense_me1/StatefulPartitionedCall!dense_me1/StatefulPartitionedCall2F
!dense_me2/StatefulPartitionedCall!dense_me2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
i
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617642

inputs
identitye
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(
IdentityIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:���������<<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<<:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
�
5__inference_transient_classifier_layer_call_fn_617030
inputs_image_input
inputs_meta_input!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	� 

unknown_12: 

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_image_inputinputs_meta_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_meta_input:c _
/
_output_shapes
:���������<<
,
_user_specified_nameinputs_image_input
�
�
)__inference_dense_c1_layer_call_fn_617726

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c1_layer_call_and_return_conditional_losses_616424p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�#
"__inference__traced_restore_618424
file_prefix8
assignvariableop_conv_0_kernel:@,
assignvariableop_1_conv_0_bias:@;
 assignvariableop_2_conv_1_kernel:@�-
assignvariableop_3_conv_1_bias:	�<
 assignvariableop_4_conv_2_kernel:��-
assignvariableop_5_conv_2_bias:	�7
)assignvariableop_6_feature_ranking_kernel:6
#assignvariableop_7_dense_me1_kernel:	�0
!assignvariableop_8_dense_me1_bias:	�7
#assignvariableop_9_dense_me2_kernel:
��1
"assignvariableop_10_dense_me2_bias:	�7
#assignvariableop_11_dense_c1_kernel:
��0
!assignvariableop_12_dense_c1_bias:	�6
#assignvariableop_13_dense_c2_kernel:	� /
!assignvariableop_14_dense_c2_bias: 3
!assignvariableop_15_output_kernel: -
assignvariableop_16_output_bias:'
assignvariableop_17_adam_iter:	 )
assignvariableop_18_adam_beta_1: )
assignvariableop_19_adam_beta_2: (
assignvariableop_20_adam_decay: #
assignvariableop_21_total: #
assignvariableop_22_count: H
:assignvariableop_23_data_augmentation_random_flip_statevar:	L
>assignvariableop_24_data_augmentation_random_rotation_statevar:	B
(assignvariableop_25_adam_conv_0_kernel_m:@4
&assignvariableop_26_adam_conv_0_bias_m:@C
(assignvariableop_27_adam_conv_1_kernel_m:@�5
&assignvariableop_28_adam_conv_1_bias_m:	�D
(assignvariableop_29_adam_conv_2_kernel_m:��5
&assignvariableop_30_adam_conv_2_bias_m:	�>
+assignvariableop_31_adam_dense_me1_kernel_m:	�8
)assignvariableop_32_adam_dense_me1_bias_m:	�?
+assignvariableop_33_adam_dense_me2_kernel_m:
��8
)assignvariableop_34_adam_dense_me2_bias_m:	�>
*assignvariableop_35_adam_dense_c1_kernel_m:
��7
(assignvariableop_36_adam_dense_c1_bias_m:	�=
*assignvariableop_37_adam_dense_c2_kernel_m:	� 6
(assignvariableop_38_adam_dense_c2_bias_m: :
(assignvariableop_39_adam_output_kernel_m: 4
&assignvariableop_40_adam_output_bias_m:B
(assignvariableop_41_adam_conv_0_kernel_v:@4
&assignvariableop_42_adam_conv_0_bias_v:@C
(assignvariableop_43_adam_conv_1_kernel_v:@�5
&assignvariableop_44_adam_conv_1_bias_v:	�D
(assignvariableop_45_adam_conv_2_kernel_v:��5
&assignvariableop_46_adam_conv_2_bias_v:	�>
+assignvariableop_47_adam_dense_me1_kernel_v:	�8
)assignvariableop_48_adam_dense_me1_bias_v:	�?
+assignvariableop_49_adam_dense_me2_kernel_v:
��8
)assignvariableop_50_adam_dense_me2_bias_v:	�>
*assignvariableop_51_adam_dense_c1_kernel_v:
��7
(assignvariableop_52_adam_dense_c1_bias_v:	�=
*assignvariableop_53_adam_dense_c2_kernel_v:	� 6
(assignvariableop_54_adam_dense_c2_bias_v: :
(assignvariableop_55_adam_output_kernel_v: 4
&assignvariableop_56_adam_output_bias_v:
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_data_augmentation/RandomFlip/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBcdata_augmentation/RandomRotation/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv_0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp)assignvariableop_6_feature_ranking_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_me1_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_me1_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_me2_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_me2_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_c1_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_c1_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_c2_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_c2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_output_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_output_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_iterIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_2Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_decayIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_data_augmentation_random_flip_statevarIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp>assignvariableop_24_data_augmentation_random_rotation_statevarIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv_0_kernel_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv_0_bias_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv_1_kernel_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv_1_bias_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv_2_kernel_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv_2_bias_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_me1_kernel_mIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_me1_bias_mIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_me2_kernel_mIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_me2_bias_mIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_c1_kernel_mIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_c1_bias_mIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_c2_kernel_mIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_c2_bias_mIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_output_kernel_mIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_output_bias_mIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_conv_0_kernel_vIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_conv_0_bias_vIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv_1_kernel_vIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_conv_1_bias_vIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_conv_2_kernel_vIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_conv_2_bias_vIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_me1_kernel_vIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_me1_bias_vIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_me2_kernel_vIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_me2_bias_vIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_c1_kernel_vIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_c1_bias_vIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_c2_kernel_vIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_c2_bias_vIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_output_kernel_vIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_output_bias_vIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
B__inference_conv_2_layer_call_and_return_conditional_losses_617857

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_data_augmentation_layer_call_fn_617404

inputs
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616294w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_616950
image_input

meta_input!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	� 

unknown_12: 

unknown_13: 

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimage_input
meta_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_616026o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:SO
'
_output_shapes
:���������
$
_user_specified_name
meta_input:\ X
/
_output_shapes
:���������<<
%
_user_specified_nameimage_input
�
�
B__inference_conv_2_layer_call_and_return_conditional_losses_616347

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_pool_1_layer_call_and_return_conditional_losses_617837

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_617647

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_616360a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv_1_layer_call_and_return_conditional_losses_617827

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
C
'__inference_pool_1_layer_call_fn_617832

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_616044�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_feature_ranking_layer_call_and_return_conditional_losses_617664

inputs
identity�
mul/yConst*
_output_shapes
:*
dtype0*U
valueLBJ"@ݖ�=�{�=��5=��=S<=�	5=p�9={�j=�Ѡ>���= J>�D�=s{�=��=z�h=|v�=T
mulMulinputsmul/y:output:0*
T0*'
_output_shapes
:���������O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv_1_layer_call_fn_617816

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_616329x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_output_layer_call_fn_617766

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_616458o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
B__inference_conv_0_layer_call_and_return_conditional_losses_617797

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������::@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������::@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
�
B__inference_conv_1_layer_call_and_return_conditional_losses_616329

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_dense_c1_layer_call_and_return_conditional_losses_616424

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_c2_layer_call_and_return_conditional_losses_616441

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_feature_ranking_layer_call_fn_617658

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_feature_ranking_layer_call_and_return_conditional_losses_616368`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
i
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616474

inputs
identitye
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(
IdentityIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:���������<<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<<:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
^
B__inference_pool_1_layer_call_and_return_conditional_losses_616044

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
D__inference_dense_c2_layer_call_and_return_conditional_losses_617757

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_me2_layer_call_and_return_conditional_losses_616398

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_me2_layer_call_fn_617693

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me2_layer_call_and_return_conditional_losses_616398p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_c1_layer_call_and_return_conditional_losses_617737

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_me2_layer_call_and_return_conditional_losses_617704

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_pool_0_layer_call_and_return_conditional_losses_617807

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
B__inference_pool_0_layer_call_and_return_conditional_losses_616032

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_616360

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_pool_2_layer_call_and_return_conditional_losses_616056

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617323
inputs_image_input
inputs_meta_input]
Odata_augmentation_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	X
Jdata_augmentation_random_rotation_stateful_uniform_rngreadandskip_resource:	?
%conv_0_conv2d_readvariableop_resource:@4
&conv_0_biasadd_readvariableop_resource:@@
%conv_1_conv2d_readvariableop_resource:@�5
&conv_1_biasadd_readvariableop_resource:	�A
%conv_2_conv2d_readvariableop_resource:��5
&conv_2_biasadd_readvariableop_resource:	�;
(dense_me1_matmul_readvariableop_resource:	�8
)dense_me1_biasadd_readvariableop_resource:	�<
(dense_me2_matmul_readvariableop_resource:
��8
)dense_me2_biasadd_readvariableop_resource:	�;
'dense_c1_matmul_readvariableop_resource:
��7
(dense_c1_biasadd_readvariableop_resource:	�:
'dense_c2_matmul_readvariableop_resource:	� 6
(dense_c2_biasadd_readvariableop_resource: 7
%output_matmul_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identity��conv_0/BiasAdd/ReadVariableOp�conv_0/Conv2D/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�Fdata_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkip�Hdata_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkip�Adata_augmentation/random_rotation/stateful_uniform/RngReadAndSkip�dense_c1/BiasAdd/ReadVariableOp�dense_c1/MatMul/ReadVariableOp�dense_c2/BiasAdd/ReadVariableOp�dense_c2/MatMul/ReadVariableOp� dense_me1/BiasAdd/ReadVariableOp�dense_me1/MatMul/ReadVariableOp� dense_me2/BiasAdd/ReadVariableOp�dense_me2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpw
&data_augmentation/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
0data_augmentation/resizing/resize/ResizeBilinearResizeBilinearinputs_image_input/data_augmentation/resizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(�
=data_augmentation/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
=data_augmentation/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
<data_augmentation/random_flip/stateful_uniform_full_int/ProdProdFdata_augmentation/random_flip/stateful_uniform_full_int/shape:output:0Fdata_augmentation/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: �
>data_augmentation/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
>data_augmentation/random_flip/stateful_uniform_full_int/Cast_1CastEdata_augmentation/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
Fdata_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipOdata_augmentation_random_flip_stateful_uniform_full_int_rngreadandskip_resourceGdata_augmentation/random_flip/stateful_uniform_full_int/Cast/x:output:0Bdata_augmentation/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:�
Kdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Mdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Mdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Edata_augmentation/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceNdata_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Tdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0Vdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Vdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
?data_augmentation/random_flip/stateful_uniform_full_int/BitcastBitcastNdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
Mdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Odata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Odata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Gdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceNdata_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Vdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Xdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Xdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
Adata_augmentation/random_flip/stateful_uniform_full_int/Bitcast_1BitcastPdata_augmentation/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0}
;data_augmentation/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :�
7data_augmentation/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2Fdata_augmentation/random_flip/stateful_uniform_full_int/shape:output:0Jdata_augmentation/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Hdata_augmentation/random_flip/stateful_uniform_full_int/Bitcast:output:0Ddata_augmentation/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	r
(data_augmentation/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R �
#data_augmentation/random_flip/stackPack@data_augmentation/random_flip/stateful_uniform_full_int:output:01data_augmentation/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:�
1data_augmentation/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
3data_augmentation/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
3data_augmentation/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
+data_augmentation/random_flip/strided_sliceStridedSlice,data_augmentation/random_flip/stack:output:0:data_augmentation/random_flip/strided_slice/stack:output:0<data_augmentation/random_flip/strided_slice/stack_1:output:0<data_augmentation/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
Qdata_augmentation/random_flip/stateless_random_flip_left_right/control_dependencyIdentityAdata_augmentation/resizing/resize/ResizeBilinear:resized_images:0*
T0*C
_class9
75loc:@data_augmentation/resizing/resize/ResizeBilinear*/
_output_shapes
:���������<<�
Ddata_augmentation/random_flip/stateless_random_flip_left_right/ShapeShapeZdata_augmentation/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::���
Rdata_augmentation/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tdata_augmentation/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tdata_augmentation/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Ldata_augmentation/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceMdata_augmentation/random_flip/stateless_random_flip_left_right/Shape:output:0[data_augmentation/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0]data_augmentation/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0]data_augmentation/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
]data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackUdata_augmentation/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:�
[data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
[data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
tdata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter4data_augmentation/random_flip/strided_slice:output:0* 
_output_shapes
::�
tdata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
pdata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2fdata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0zdata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0~data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0}data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
[data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubddata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0ddata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
[data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulydata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0_data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
Wdata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2_data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0ddata_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:����������
Ndata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
Ndata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Ndata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Ldata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shapePackUdata_augmentation/random_flip/stateless_random_flip_left_right/strided_slice:output:0Wdata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Wdata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Wdata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fdata_augmentation/random_flip/stateless_random_flip_left_right/ReshapeReshape[data_augmentation/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Udata_augmentation/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
Ddata_augmentation/random_flip/stateless_random_flip_left_right/RoundRoundOdata_augmentation/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:����������
Mdata_augmentation/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
Hdata_augmentation/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Zdata_augmentation/random_flip/stateless_random_flip_left_right/control_dependency:output:0Vdata_augmentation/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:���������<<�
Bdata_augmentation/random_flip/stateless_random_flip_left_right/mulMulHdata_augmentation/random_flip/stateless_random_flip_left_right/Round:y:0Qdata_augmentation/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:���������<<�
Ddata_augmentation/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Bdata_augmentation/random_flip/stateless_random_flip_left_right/subSubMdata_augmentation/random_flip/stateless_random_flip_left_right/sub/x:output:0Hdata_augmentation/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:����������
Ddata_augmentation/random_flip/stateless_random_flip_left_right/mul_1MulFdata_augmentation/random_flip/stateless_random_flip_left_right/sub:z:0Zdata_augmentation/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:���������<<�
Bdata_augmentation/random_flip/stateless_random_flip_left_right/addAddV2Fdata_augmentation/random_flip/stateless_random_flip_left_right/mul:z:0Hdata_augmentation/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:���������<<�
?data_augmentation/random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
?data_augmentation/random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
>data_augmentation/random_flip/stateful_uniform_full_int_1/ProdProdHdata_augmentation/random_flip/stateful_uniform_full_int_1/shape:output:0Hdata_augmentation/random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: �
@data_augmentation/random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
@data_augmentation/random_flip/stateful_uniform_full_int_1/Cast_1CastGdata_augmentation/random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
Hdata_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkipOdata_augmentation_random_flip_stateful_uniform_full_int_rngreadandskip_resourceIdata_augmentation/random_flip/stateful_uniform_full_int_1/Cast/x:output:0Ddata_augmentation/random_flip/stateful_uniform_full_int_1/Cast_1:y:0G^data_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:�
Mdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Odata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Odata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Gdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlicePdata_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Vdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Xdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Xdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
Adata_augmentation/random_flip/stateful_uniform_full_int_1/BitcastBitcastPdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
Odata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Qdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Qdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Idata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlicePdata_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Xdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Zdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Zdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
Cdata_augmentation/random_flip/stateful_uniform_full_int_1/Bitcast_1BitcastRdata_augmentation/random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
=data_augmentation/random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :�
9data_augmentation/random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV2Hdata_augmentation/random_flip/stateful_uniform_full_int_1/shape:output:0Ldata_augmentation/random_flip/stateful_uniform_full_int_1/Bitcast_1:output:0Jdata_augmentation/random_flip/stateful_uniform_full_int_1/Bitcast:output:0Fdata_augmentation/random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	t
*data_augmentation/random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R �
%data_augmentation/random_flip/stack_1PackBdata_augmentation/random_flip/stateful_uniform_full_int_1:output:03data_augmentation/random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:�
3data_augmentation/random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
5data_augmentation/random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
5data_augmentation/random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
-data_augmentation/random_flip/strided_slice_1StridedSlice.data_augmentation/random_flip/stack_1:output:0<data_augmentation/random_flip/strided_slice_1/stack:output:0>data_augmentation/random_flip/strided_slice_1/stack_1:output:0>data_augmentation/random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
Ndata_augmentation/random_flip/stateless_random_flip_up_down/control_dependencyIdentityFdata_augmentation/random_flip/stateless_random_flip_left_right/add:z:0*
T0*U
_classK
IGloc:@data_augmentation/random_flip/stateless_random_flip_left_right/add*/
_output_shapes
:���������<<�
Adata_augmentation/random_flip/stateless_random_flip_up_down/ShapeShapeWdata_augmentation/random_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::���
Odata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Qdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Qdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Idata_augmentation/random_flip/stateless_random_flip_up_down/strided_sliceStridedSliceJdata_augmentation/random_flip/stateless_random_flip_up_down/Shape:output:0Xdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Zdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Zdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Zdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePackRdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:�
Xdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Xdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
qdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter6data_augmentation/random_flip/strided_slice_1:output:0* 
_output_shapes
::�
qdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
mdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2cdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0wdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0{data_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0zdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
Xdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubadata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0adata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
Xdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMulvdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0\data_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
Tdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2\data_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0adata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:����������
Kdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
Kdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Kdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Idata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shapePackRdata_augmentation/random_flip/stateless_random_flip_up_down/strided_slice:output:0Tdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Tdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Tdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Cdata_augmentation/random_flip/stateless_random_flip_up_down/ReshapeReshapeXdata_augmentation/random_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0Rdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
Adata_augmentation/random_flip/stateless_random_flip_up_down/RoundRoundLdata_augmentation/random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:����������
Jdata_augmentation/random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
Edata_augmentation/random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Wdata_augmentation/random_flip/stateless_random_flip_up_down/control_dependency:output:0Sdata_augmentation/random_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*/
_output_shapes
:���������<<�
?data_augmentation/random_flip/stateless_random_flip_up_down/mulMulEdata_augmentation/random_flip/stateless_random_flip_up_down/Round:y:0Ndata_augmentation/random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*/
_output_shapes
:���������<<�
Adata_augmentation/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
?data_augmentation/random_flip/stateless_random_flip_up_down/subSubJdata_augmentation/random_flip/stateless_random_flip_up_down/sub/x:output:0Edata_augmentation/random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:����������
Adata_augmentation/random_flip/stateless_random_flip_up_down/mul_1MulCdata_augmentation/random_flip/stateless_random_flip_up_down/sub:z:0Wdata_augmentation/random_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*/
_output_shapes
:���������<<�
?data_augmentation/random_flip/stateless_random_flip_up_down/addAddV2Cdata_augmentation/random_flip/stateless_random_flip_up_down/mul:z:0Edata_augmentation/random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*/
_output_shapes
:���������<<�
'data_augmentation/random_rotation/ShapeShapeCdata_augmentation/random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
::��
5data_augmentation/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7data_augmentation/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7data_augmentation/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/data_augmentation/random_rotation/strided_sliceStridedSlice0data_augmentation/random_rotation/Shape:output:0>data_augmentation/random_rotation/strided_slice/stack:output:0@data_augmentation/random_rotation/strided_slice/stack_1:output:0@data_augmentation/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7data_augmentation/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
9data_augmentation/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
9data_augmentation/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1data_augmentation/random_rotation/strided_slice_1StridedSlice0data_augmentation/random_rotation/Shape:output:0@data_augmentation/random_rotation/strided_slice_1/stack:output:0Bdata_augmentation/random_rotation/strided_slice_1/stack_1:output:0Bdata_augmentation/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
&data_augmentation/random_rotation/CastCast:data_augmentation/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: �
7data_augmentation/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
9data_augmentation/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
9data_augmentation/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1data_augmentation/random_rotation/strided_slice_2StridedSlice0data_augmentation/random_rotation/Shape:output:0@data_augmentation/random_rotation/strided_slice_2/stack:output:0Bdata_augmentation/random_rotation/strided_slice_2/stack_1:output:0Bdata_augmentation/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
(data_augmentation/random_rotation/Cast_1Cast:data_augmentation/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: �
8data_augmentation/random_rotation/stateful_uniform/shapePack8data_augmentation/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:{
6data_augmentation/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *���{
6data_augmentation/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
8data_augmentation/random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7data_augmentation/random_rotation/stateful_uniform/ProdProdAdata_augmentation/random_rotation/stateful_uniform/shape:output:0Adata_augmentation/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: {
9data_augmentation/random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
9data_augmentation/random_rotation/stateful_uniform/Cast_1Cast@data_augmentation/random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
Adata_augmentation/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkipJdata_augmentation_random_rotation_stateful_uniform_rngreadandskip_resourceBdata_augmentation/random_rotation/stateful_uniform/Cast/x:output:0=data_augmentation/random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:�
Fdata_augmentation/random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Hdata_augmentation/random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Hdata_augmentation/random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@data_augmentation/random_rotation/stateful_uniform/strided_sliceStridedSliceIdata_augmentation/random_rotation/stateful_uniform/RngReadAndSkip:value:0Odata_augmentation/random_rotation/stateful_uniform/strided_slice/stack:output:0Qdata_augmentation/random_rotation/stateful_uniform/strided_slice/stack_1:output:0Qdata_augmentation/random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
:data_augmentation/random_rotation/stateful_uniform/BitcastBitcastIdata_augmentation/random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
Hdata_augmentation/random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Jdata_augmentation/random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jdata_augmentation/random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bdata_augmentation/random_rotation/stateful_uniform/strided_slice_1StridedSliceIdata_augmentation/random_rotation/stateful_uniform/RngReadAndSkip:value:0Qdata_augmentation/random_rotation/stateful_uniform/strided_slice_1/stack:output:0Sdata_augmentation/random_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Sdata_augmentation/random_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
<data_augmentation/random_rotation/stateful_uniform/Bitcast_1BitcastKdata_augmentation/random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0�
Odata_augmentation/random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
Kdata_augmentation/random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Adata_augmentation/random_rotation/stateful_uniform/shape:output:0Edata_augmentation/random_rotation/stateful_uniform/Bitcast_1:output:0Cdata_augmentation/random_rotation/stateful_uniform/Bitcast:output:0Xdata_augmentation/random_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
6data_augmentation/random_rotation/stateful_uniform/subSub?data_augmentation/random_rotation/stateful_uniform/max:output:0?data_augmentation/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: �
6data_augmentation/random_rotation/stateful_uniform/mulMulTdata_augmentation/random_rotation/stateful_uniform/StatelessRandomUniformV2:output:0:data_augmentation/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:����������
2data_augmentation/random_rotation/stateful_uniformAddV2:data_augmentation/random_rotation/stateful_uniform/mul:z:0?data_augmentation/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������|
7data_augmentation/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
5data_augmentation/random_rotation/rotation_matrix/subSub,data_augmentation/random_rotation/Cast_1:y:0@data_augmentation/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: �
5data_augmentation/random_rotation/rotation_matrix/CosCos6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������~
9data_augmentation/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7data_augmentation/random_rotation/rotation_matrix/sub_1Sub,data_augmentation/random_rotation/Cast_1:y:0Bdata_augmentation/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: �
5data_augmentation/random_rotation/rotation_matrix/mulMul9data_augmentation/random_rotation/rotation_matrix/Cos:y:0;data_augmentation/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:����������
5data_augmentation/random_rotation/rotation_matrix/SinSin6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������~
9data_augmentation/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7data_augmentation/random_rotation/rotation_matrix/sub_2Sub*data_augmentation/random_rotation/Cast:y:0Bdata_augmentation/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: �
7data_augmentation/random_rotation/rotation_matrix/mul_1Mul9data_augmentation/random_rotation/rotation_matrix/Sin:y:0;data_augmentation/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:����������
7data_augmentation/random_rotation/rotation_matrix/sub_3Sub9data_augmentation/random_rotation/rotation_matrix/mul:z:0;data_augmentation/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:����������
7data_augmentation/random_rotation/rotation_matrix/sub_4Sub9data_augmentation/random_rotation/rotation_matrix/sub:z:0;data_augmentation/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:����������
;data_augmentation/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
9data_augmentation/random_rotation/rotation_matrix/truedivRealDiv;data_augmentation/random_rotation/rotation_matrix/sub_4:z:0Ddata_augmentation/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������~
9data_augmentation/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7data_augmentation/random_rotation/rotation_matrix/sub_5Sub*data_augmentation/random_rotation/Cast:y:0Bdata_augmentation/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: �
7data_augmentation/random_rotation/rotation_matrix/Sin_1Sin6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������~
9data_augmentation/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7data_augmentation/random_rotation/rotation_matrix/sub_6Sub,data_augmentation/random_rotation/Cast_1:y:0Bdata_augmentation/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: �
7data_augmentation/random_rotation/rotation_matrix/mul_2Mul;data_augmentation/random_rotation/rotation_matrix/Sin_1:y:0;data_augmentation/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:����������
7data_augmentation/random_rotation/rotation_matrix/Cos_1Cos6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������~
9data_augmentation/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7data_augmentation/random_rotation/rotation_matrix/sub_7Sub*data_augmentation/random_rotation/Cast:y:0Bdata_augmentation/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: �
7data_augmentation/random_rotation/rotation_matrix/mul_3Mul;data_augmentation/random_rotation/rotation_matrix/Cos_1:y:0;data_augmentation/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:����������
5data_augmentation/random_rotation/rotation_matrix/addAddV2;data_augmentation/random_rotation/rotation_matrix/mul_2:z:0;data_augmentation/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:����������
7data_augmentation/random_rotation/rotation_matrix/sub_8Sub;data_augmentation/random_rotation/rotation_matrix/sub_5:z:09data_augmentation/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:����������
=data_augmentation/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
;data_augmentation/random_rotation/rotation_matrix/truediv_1RealDiv;data_augmentation/random_rotation/rotation_matrix/sub_8:z:0Fdata_augmentation/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:����������
7data_augmentation/random_rotation/rotation_matrix/ShapeShape6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::���
Edata_augmentation/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?data_augmentation/random_rotation/rotation_matrix/strided_sliceStridedSlice@data_augmentation/random_rotation/rotation_matrix/Shape:output:0Ndata_augmentation/random_rotation/rotation_matrix/strided_slice/stack:output:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice/stack_1:output:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7data_augmentation/random_rotation/rotation_matrix/Cos_2Cos6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Adata_augmentation/random_rotation/rotation_matrix/strided_slice_1StridedSlice;data_augmentation/random_rotation/rotation_matrix/Cos_2:y:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice_1/stack:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
7data_augmentation/random_rotation/rotation_matrix/Sin_2Sin6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Adata_augmentation/random_rotation/rotation_matrix/strided_slice_2StridedSlice;data_augmentation/random_rotation/rotation_matrix/Sin_2:y:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice_2/stack:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
5data_augmentation/random_rotation/rotation_matrix/NegNegJdata_augmentation/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:����������
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Adata_augmentation/random_rotation/rotation_matrix/strided_slice_3StridedSlice=data_augmentation/random_rotation/rotation_matrix/truediv:z:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice_3/stack:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
7data_augmentation/random_rotation/rotation_matrix/Sin_3Sin6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Adata_augmentation/random_rotation/rotation_matrix/strided_slice_4StridedSlice;data_augmentation/random_rotation/rotation_matrix/Sin_3:y:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice_4/stack:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
7data_augmentation/random_rotation/rotation_matrix/Cos_3Cos6data_augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Adata_augmentation/random_rotation/rotation_matrix/strided_slice_5StridedSlice;data_augmentation/random_rotation/rotation_matrix/Cos_3:y:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice_5/stack:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
Gdata_augmentation/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
Idata_augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Adata_augmentation/random_rotation/rotation_matrix/strided_slice_6StridedSlice?data_augmentation/random_rotation/rotation_matrix/truediv_1:z:0Pdata_augmentation/random_rotation/rotation_matrix/strided_slice_6/stack:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0Rdata_augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
@data_augmentation/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
>data_augmentation/random_rotation/rotation_matrix/zeros/packedPackHdata_augmentation/random_rotation/rotation_matrix/strided_slice:output:0Idata_augmentation/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
=data_augmentation/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
7data_augmentation/random_rotation/rotation_matrix/zerosFillGdata_augmentation/random_rotation/rotation_matrix/zeros/packed:output:0Fdata_augmentation/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������
=data_augmentation/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
8data_augmentation/random_rotation/rotation_matrix/concatConcatV2Jdata_augmentation/random_rotation/rotation_matrix/strided_slice_1:output:09data_augmentation/random_rotation/rotation_matrix/Neg:y:0Jdata_augmentation/random_rotation/rotation_matrix/strided_slice_3:output:0Jdata_augmentation/random_rotation/rotation_matrix/strided_slice_4:output:0Jdata_augmentation/random_rotation/rotation_matrix/strided_slice_5:output:0Jdata_augmentation/random_rotation/rotation_matrix/strided_slice_6:output:0@data_augmentation/random_rotation/rotation_matrix/zeros:output:0Fdata_augmentation/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
1data_augmentation/random_rotation/transform/ShapeShapeCdata_augmentation/random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
::���
?data_augmentation/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Adata_augmentation/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Adata_augmentation/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9data_augmentation/random_rotation/transform/strided_sliceStridedSlice:data_augmentation/random_rotation/transform/Shape:output:0Hdata_augmentation/random_rotation/transform/strided_slice/stack:output:0Jdata_augmentation/random_rotation/transform/strided_slice/stack_1:output:0Jdata_augmentation/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:{
6data_augmentation/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Fdata_augmentation/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Cdata_augmentation/random_flip/stateless_random_flip_up_down/add:z:0Adata_augmentation/random_rotation/rotation_matrix/concat:output:0Bdata_augmentation/random_rotation/transform/strided_slice:output:0?data_augmentation/random_rotation/transform/fill_value:output:0*/
_output_shapes
:���������<<*
dtype0*
	fill_mode	NEAREST*
interpolation
BILINEAR�
conv_0/Conv2D/ReadVariableOpReadVariableOp%conv_0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv_0/Conv2DConv2D[data_augmentation/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0$conv_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@*
paddingVALID*
strides
�
conv_0/BiasAdd/ReadVariableOpReadVariableOp&conv_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv_0/BiasAddBiasAddconv_0/Conv2D:output:0%conv_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@f
conv_0/ReluReluconv_0/BiasAdd:output:0*
T0*/
_output_shapes
:���������::@�
pool_0/MaxPoolMaxPoolconv_0/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv_1/Conv2DConv2Dpool_0/MaxPool:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
pool_1/MaxPoolMaxPoolconv_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv_2/Conv2DConv2Dpool_1/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
pool_2/MaxPoolMaxPoolconv_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ~
flatten/ReshapeReshapepool_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
feature_ranking/mul/yConst*
_output_shapes
:*
dtype0*U
valueLBJ"@ݖ�=�{�=��5=��=S<=�	5=p�9={�j=�Ѡ>���= J>�D�=s{�=��=z�h=|v�=
feature_ranking/mulMulinputs_meta_inputfeature_ranking/mul/y:output:0*
T0*'
_output_shapes
:����������
dense_me1/MatMul/ReadVariableOpReadVariableOp(dense_me1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_me1/MatMulMatMulfeature_ranking/mul:z:0'dense_me1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_me1/BiasAdd/ReadVariableOpReadVariableOp)dense_me1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_me1/BiasAddBiasAdddense_me1/MatMul:product:0(dense_me1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_me1/ReluReludense_me1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_me2/MatMul/ReadVariableOpReadVariableOp(dense_me2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_me2/MatMulMatMuldense_me1/Relu:activations:0'dense_me2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_me2/BiasAdd/ReadVariableOpReadVariableOp)dense_me2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_me2/BiasAddBiasAdddense_me2/MatMul:product:0(dense_me2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_me2/ReluReludense_me2/BiasAdd:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0dense_me2/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_c1/MatMul/ReadVariableOpReadVariableOp'dense_c1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_c1/MatMulMatMulconcatenate/concat:output:0&dense_c1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_c1/BiasAdd/ReadVariableOpReadVariableOp(dense_c1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_c1/BiasAddBiasAdddense_c1/MatMul:product:0'dense_c1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_c1/ReluReludense_c1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_c2/MatMul/ReadVariableOpReadVariableOp'dense_c2_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_c2/MatMulMatMuldense_c1/Relu:activations:0&dense_c2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_c2/BiasAdd/ReadVariableOpReadVariableOp(dense_c2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_c2/BiasAddBiasAdddense_c2/MatMul:product:0'dense_c2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_c2/ReluReludense_c2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output/MatMulMatMuldense_c2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������g
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv_0/BiasAdd/ReadVariableOp^conv_0/Conv2D/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOpG^data_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkipI^data_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkipB^data_augmentation/random_rotation/stateful_uniform/RngReadAndSkip ^dense_c1/BiasAdd/ReadVariableOp^dense_c1/MatMul/ReadVariableOp ^dense_c2/BiasAdd/ReadVariableOp^dense_c2/MatMul/ReadVariableOp!^dense_me1/BiasAdd/ReadVariableOp ^dense_me1/MatMul/ReadVariableOp!^dense_me2/BiasAdd/ReadVariableOp ^dense_me2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������<<:���������: : : : : : : : : : : : : : : : : : 2>
conv_0/BiasAdd/ReadVariableOpconv_0/BiasAdd/ReadVariableOp2<
conv_0/Conv2D/ReadVariableOpconv_0/Conv2D/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2�
Fdata_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkipFdata_augmentation/random_flip/stateful_uniform_full_int/RngReadAndSkip2�
Hdata_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkipHdata_augmentation/random_flip/stateful_uniform_full_int_1/RngReadAndSkip2�
Adata_augmentation/random_rotation/stateful_uniform/RngReadAndSkipAdata_augmentation/random_rotation/stateful_uniform/RngReadAndSkip2B
dense_c1/BiasAdd/ReadVariableOpdense_c1/BiasAdd/ReadVariableOp2@
dense_c1/MatMul/ReadVariableOpdense_c1/MatMul/ReadVariableOp2B
dense_c2/BiasAdd/ReadVariableOpdense_c2/BiasAdd/ReadVariableOp2@
dense_c2/MatMul/ReadVariableOpdense_c2/MatMul/ReadVariableOp2D
 dense_me1/BiasAdd/ReadVariableOp dense_me1/BiasAdd/ReadVariableOp2B
dense_me1/MatMul/ReadVariableOpdense_me1/MatMul/ReadVariableOp2D
 dense_me2/BiasAdd/ReadVariableOp dense_me2/BiasAdd/ReadVariableOp2B
dense_me2/MatMul/ReadVariableOpdense_me2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_meta_input:c _
/
_output_shapes
:���������<<
,
_user_specified_nameinputs_image_input
�
s
G__inference_concatenate_layer_call_and_return_conditional_losses_617717
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs_0
�A
�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616465
image_input

meta_input&
data_augmentation_616295:	&
data_augmentation_616297:	'
conv_0_616312:@
conv_0_616314:@(
conv_1_616330:@�
conv_1_616332:	�)
conv_2_616348:��
conv_2_616350:	�#
dense_me1_616382:	�
dense_me1_616384:	�$
dense_me2_616399:
��
dense_me2_616401:	�#
dense_c1_616425:
��
dense_c1_616427:	�"
dense_c2_616442:	� 
dense_c2_616444: 
output_616459: 
output_616461:
identity��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�)data_augmentation/StatefulPartitionedCall� dense_c1/StatefulPartitionedCall� dense_c2/StatefulPartitionedCall�!dense_me1/StatefulPartitionedCall�!dense_me2/StatefulPartitionedCall�output/StatefulPartitionedCall�
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallimage_inputdata_augmentation_616295data_augmentation_616297*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616294�
conv_0/StatefulPartitionedCallStatefulPartitionedCall2data_augmentation/StatefulPartitionedCall:output:0conv_0_616312conv_0_616314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������::@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_616311�
pool_0/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_0_layer_call_and_return_conditional_losses_616032�
conv_1/StatefulPartitionedCallStatefulPartitionedCallpool_0/PartitionedCall:output:0conv_1_616330conv_1_616332*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_616329�
pool_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_616044�
conv_2/StatefulPartitionedCallStatefulPartitionedCallpool_1/PartitionedCall:output:0conv_2_616348conv_2_616350*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_616347�
pool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_616056�
flatten/PartitionedCallPartitionedCallpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_616360�
feature_ranking/PartitionedCallPartitionedCall
meta_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_feature_ranking_layer_call_and_return_conditional_losses_616368�
!dense_me1/StatefulPartitionedCallStatefulPartitionedCall(feature_ranking/PartitionedCall:output:0dense_me1_616382dense_me1_616384*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me1_layer_call_and_return_conditional_losses_616381�
!dense_me2/StatefulPartitionedCallStatefulPartitionedCall*dense_me1/StatefulPartitionedCall:output:0dense_me2_616399dense_me2_616401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me2_layer_call_and_return_conditional_losses_616398�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*dense_me2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_616411�
 dense_c1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_c1_616425dense_c1_616427*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c1_layer_call_and_return_conditional_losses_616424�
 dense_c2/StatefulPartitionedCallStatefulPartitionedCall)dense_c1/StatefulPartitionedCall:output:0dense_c2_616442dense_c2_616444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c2_layer_call_and_return_conditional_losses_616441�
output/StatefulPartitionedCallStatefulPartitionedCall)dense_c2/StatefulPartitionedCall:output:0output_616459output_616461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_616458v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall*^data_augmentation/StatefulPartitionedCall!^dense_c1/StatefulPartitionedCall!^dense_c2/StatefulPartitionedCall"^dense_me1/StatefulPartitionedCall"^dense_me2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������<<:���������: : : : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_c1/StatefulPartitionedCall dense_c1/StatefulPartitionedCall2D
 dense_c2/StatefulPartitionedCall dense_c2/StatefulPartitionedCall2F
!dense_me1/StatefulPartitionedCall!dense_me1/StatefulPartitionedCall2F
!dense_me2/StatefulPartitionedCall!dense_me2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:SO
'
_output_shapes
:���������
$
_user_specified_name
meta_input:\ X
/
_output_shapes
:���������<<
%
_user_specified_nameimage_input
�
C
'__inference_pool_0_layer_call_fn_617802

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_0_layer_call_and_return_conditional_losses_616032�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_dense_c2_layer_call_fn_617746

inputs
unknown:	� 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_c2_layer_call_and_return_conditional_losses_616441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_data_augmentation_layer_call_fn_617409

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616474h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������<<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<<:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
�
B__inference_conv_0_layer_call_and_return_conditional_losses_616311

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������::@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������::@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
C
'__inference_pool_2_layer_call_fn_617862

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_616056�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
B__inference_pool_2_layer_call_and_return_conditional_losses_617867

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
ݙ
�4
__inference__traced_save_618243
file_prefix>
$read_disablecopyonread_conv_0_kernel:@2
$read_1_disablecopyonread_conv_0_bias:@A
&read_2_disablecopyonread_conv_1_kernel:@�3
$read_3_disablecopyonread_conv_1_bias:	�B
&read_4_disablecopyonread_conv_2_kernel:��3
$read_5_disablecopyonread_conv_2_bias:	�=
/read_6_disablecopyonread_feature_ranking_kernel:<
)read_7_disablecopyonread_dense_me1_kernel:	�6
'read_8_disablecopyonread_dense_me1_bias:	�=
)read_9_disablecopyonread_dense_me2_kernel:
��7
(read_10_disablecopyonread_dense_me2_bias:	�=
)read_11_disablecopyonread_dense_c1_kernel:
��6
'read_12_disablecopyonread_dense_c1_bias:	�<
)read_13_disablecopyonread_dense_c2_kernel:	� 5
'read_14_disablecopyonread_dense_c2_bias: 9
'read_15_disablecopyonread_output_kernel: 3
%read_16_disablecopyonread_output_bias:-
#read_17_disablecopyonread_adam_iter:	 /
%read_18_disablecopyonread_adam_beta_1: /
%read_19_disablecopyonread_adam_beta_2: .
$read_20_disablecopyonread_adam_decay: )
read_21_disablecopyonread_total: )
read_22_disablecopyonread_count: N
@read_23_disablecopyonread_data_augmentation_random_flip_statevar:	R
Dread_24_disablecopyonread_data_augmentation_random_rotation_statevar:	H
.read_25_disablecopyonread_adam_conv_0_kernel_m:@:
,read_26_disablecopyonread_adam_conv_0_bias_m:@I
.read_27_disablecopyonread_adam_conv_1_kernel_m:@�;
,read_28_disablecopyonread_adam_conv_1_bias_m:	�J
.read_29_disablecopyonread_adam_conv_2_kernel_m:��;
,read_30_disablecopyonread_adam_conv_2_bias_m:	�D
1read_31_disablecopyonread_adam_dense_me1_kernel_m:	�>
/read_32_disablecopyonread_adam_dense_me1_bias_m:	�E
1read_33_disablecopyonread_adam_dense_me2_kernel_m:
��>
/read_34_disablecopyonread_adam_dense_me2_bias_m:	�D
0read_35_disablecopyonread_adam_dense_c1_kernel_m:
��=
.read_36_disablecopyonread_adam_dense_c1_bias_m:	�C
0read_37_disablecopyonread_adam_dense_c2_kernel_m:	� <
.read_38_disablecopyonread_adam_dense_c2_bias_m: @
.read_39_disablecopyonread_adam_output_kernel_m: :
,read_40_disablecopyonread_adam_output_bias_m:H
.read_41_disablecopyonread_adam_conv_0_kernel_v:@:
,read_42_disablecopyonread_adam_conv_0_bias_v:@I
.read_43_disablecopyonread_adam_conv_1_kernel_v:@�;
,read_44_disablecopyonread_adam_conv_1_bias_v:	�J
.read_45_disablecopyonread_adam_conv_2_kernel_v:��;
,read_46_disablecopyonread_adam_conv_2_bias_v:	�D
1read_47_disablecopyonread_adam_dense_me1_kernel_v:	�>
/read_48_disablecopyonread_adam_dense_me1_bias_v:	�E
1read_49_disablecopyonread_adam_dense_me2_kernel_v:
��>
/read_50_disablecopyonread_adam_dense_me2_bias_v:	�D
0read_51_disablecopyonread_adam_dense_c1_kernel_v:
��=
.read_52_disablecopyonread_adam_dense_c1_bias_v:	�C
0read_53_disablecopyonread_adam_dense_c2_kernel_v:	� <
.read_54_disablecopyonread_adam_dense_c2_bias_v: @
.read_55_disablecopyonread_adam_output_kernel_v: :
,read_56_disablecopyonread_adam_output_bias_v:
savev2_const
identity_115��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv_0_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv_0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv_0_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv_0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@z
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_conv_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_conv_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0v

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�l

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�x
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_conv_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_conv_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_conv_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_conv_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0w

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��m

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*(
_output_shapes
:��x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_conv_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_conv_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead/read_6_disablecopyonread_feature_ranking_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp/read_6_disablecopyonread_feature_ranking_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_dense_me1_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_dense_me1_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_me1_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_me1_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_dense_me2_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_dense_me2_kernel^Read_9/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��}
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_dense_me2_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_dense_me2_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_dense_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_dense_c1_kernel^Read_11/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_dense_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_dense_c1_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_dense_c2_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_dense_c2_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	� |
Read_14/DisableCopyOnReadDisableCopyOnRead'read_14_disablecopyonread_dense_c2_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp'read_14_disablecopyonread_dense_c2_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_output_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

: z
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_output_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_output_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_17/DisableCopyOnReadDisableCopyOnRead#read_17_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp#read_17_disablecopyonread_adam_iter^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_adam_beta_1^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_adam_beta_2^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_20/DisableCopyOnReadDisableCopyOnRead$read_20_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp$read_20_disablecopyonread_adam_decay^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_21/DisableCopyOnReadDisableCopyOnReadread_21_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpread_21_disablecopyonread_total^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_count^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_23/DisableCopyOnReadDisableCopyOnRead@read_23_disablecopyonread_data_augmentation_random_flip_statevar"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp@read_23_disablecopyonread_data_augmentation_random_flip_statevar^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnReadDread_24_disablecopyonread_data_augmentation_random_rotation_statevar"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpDread_24_disablecopyonread_data_augmentation_random_rotation_statevar^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_conv_0_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_conv_0_kernel_m^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_26/DisableCopyOnReadDisableCopyOnRead,read_26_disablecopyonread_adam_conv_0_bias_m"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp,read_26_disablecopyonread_adam_conv_0_bias_m^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_conv_1_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_conv_1_kernel_m^Read_27/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_28/DisableCopyOnReadDisableCopyOnRead,read_28_disablecopyonread_adam_conv_1_bias_m"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp,read_28_disablecopyonread_adam_conv_1_bias_m^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_conv_2_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_conv_2_kernel_m^Read_29/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_30/DisableCopyOnReadDisableCopyOnRead,read_30_disablecopyonread_adam_conv_2_bias_m"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp,read_30_disablecopyonread_adam_conv_2_bias_m^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead1read_31_disablecopyonread_adam_dense_me1_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp1read_31_disablecopyonread_adam_dense_me1_kernel_m^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_32/DisableCopyOnReadDisableCopyOnRead/read_32_disablecopyonread_adam_dense_me1_bias_m"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp/read_32_disablecopyonread_adam_dense_me1_bias_m^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead1read_33_disablecopyonread_adam_dense_me2_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp1read_33_disablecopyonread_adam_dense_me2_kernel_m^Read_33/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_34/DisableCopyOnReadDisableCopyOnRead/read_34_disablecopyonread_adam_dense_me2_bias_m"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp/read_34_disablecopyonread_adam_dense_me2_bias_m^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_dense_c1_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_dense_c1_kernel_m^Read_35/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_dense_c1_bias_m"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_dense_c1_bias_m^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnRead0read_37_disablecopyonread_adam_dense_c2_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp0read_37_disablecopyonread_adam_dense_c2_kernel_m^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_38/DisableCopyOnReadDisableCopyOnRead.read_38_disablecopyonread_adam_dense_c2_bias_m"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp.read_38_disablecopyonread_adam_dense_c2_bias_m^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_39/DisableCopyOnReadDisableCopyOnRead.read_39_disablecopyonread_adam_output_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp.read_39_disablecopyonread_adam_output_kernel_m^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_40/DisableCopyOnReadDisableCopyOnRead,read_40_disablecopyonread_adam_output_bias_m"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp,read_40_disablecopyonread_adam_output_bias_m^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_conv_0_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_conv_0_kernel_v^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_42/DisableCopyOnReadDisableCopyOnRead,read_42_disablecopyonread_adam_conv_0_bias_v"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp,read_42_disablecopyonread_adam_conv_0_bias_v^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_43/DisableCopyOnReadDisableCopyOnRead.read_43_disablecopyonread_adam_conv_1_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp.read_43_disablecopyonread_adam_conv_1_kernel_v^Read_43/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_44/DisableCopyOnReadDisableCopyOnRead,read_44_disablecopyonread_adam_conv_1_bias_v"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp,read_44_disablecopyonread_adam_conv_1_bias_v^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_conv_2_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_conv_2_kernel_v^Read_45/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_46/DisableCopyOnReadDisableCopyOnRead,read_46_disablecopyonread_adam_conv_2_bias_v"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp,read_46_disablecopyonread_adam_conv_2_bias_v^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnRead1read_47_disablecopyonread_adam_dense_me1_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp1read_47_disablecopyonread_adam_dense_me1_kernel_v^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_48/DisableCopyOnReadDisableCopyOnRead/read_48_disablecopyonread_adam_dense_me1_bias_v"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp/read_48_disablecopyonread_adam_dense_me1_bias_v^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead1read_49_disablecopyonread_adam_dense_me2_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp1read_49_disablecopyonread_adam_dense_me2_kernel_v^Read_49/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_50/DisableCopyOnReadDisableCopyOnRead/read_50_disablecopyonread_adam_dense_me2_bias_v"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp/read_50_disablecopyonread_adam_dense_me2_bias_v^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead0read_51_disablecopyonread_adam_dense_c1_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp0read_51_disablecopyonread_adam_dense_c1_kernel_v^Read_51/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_52/DisableCopyOnReadDisableCopyOnRead.read_52_disablecopyonread_adam_dense_c1_bias_v"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp.read_52_disablecopyonread_adam_dense_c1_bias_v^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_dense_c2_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_dense_c2_kernel_v^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0q
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� h
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_adam_dense_c2_bias_v"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_adam_dense_c2_bias_v^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_55/DisableCopyOnReadDisableCopyOnRead.read_55_disablecopyonread_adam_output_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp.read_55_disablecopyonread_adam_output_kernel_v^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_56/DisableCopyOnReadDisableCopyOnRead,read_56_disablecopyonread_adam_output_bias_v"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp,read_56_disablecopyonread_adam_output_bias_v^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_data_augmentation/RandomFlip/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBcdata_augmentation/RandomRotation/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *H
dtypes>
<2:			�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_114Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_115IdentityIdentity_114:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_115Identity_115:output:0*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp::

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�u
�
!__inference__wrapped_model_616026
image_input

meta_inputT
:transient_classifier_conv_0_conv2d_readvariableop_resource:@I
;transient_classifier_conv_0_biasadd_readvariableop_resource:@U
:transient_classifier_conv_1_conv2d_readvariableop_resource:@�J
;transient_classifier_conv_1_biasadd_readvariableop_resource:	�V
:transient_classifier_conv_2_conv2d_readvariableop_resource:��J
;transient_classifier_conv_2_biasadd_readvariableop_resource:	�P
=transient_classifier_dense_me1_matmul_readvariableop_resource:	�M
>transient_classifier_dense_me1_biasadd_readvariableop_resource:	�Q
=transient_classifier_dense_me2_matmul_readvariableop_resource:
��M
>transient_classifier_dense_me2_biasadd_readvariableop_resource:	�P
<transient_classifier_dense_c1_matmul_readvariableop_resource:
��L
=transient_classifier_dense_c1_biasadd_readvariableop_resource:	�O
<transient_classifier_dense_c2_matmul_readvariableop_resource:	� K
=transient_classifier_dense_c2_biasadd_readvariableop_resource: L
:transient_classifier_output_matmul_readvariableop_resource: I
;transient_classifier_output_biasadd_readvariableop_resource:
identity��2transient_classifier/conv_0/BiasAdd/ReadVariableOp�1transient_classifier/conv_0/Conv2D/ReadVariableOp�2transient_classifier/conv_1/BiasAdd/ReadVariableOp�1transient_classifier/conv_1/Conv2D/ReadVariableOp�2transient_classifier/conv_2/BiasAdd/ReadVariableOp�1transient_classifier/conv_2/Conv2D/ReadVariableOp�4transient_classifier/dense_c1/BiasAdd/ReadVariableOp�3transient_classifier/dense_c1/MatMul/ReadVariableOp�4transient_classifier/dense_c2/BiasAdd/ReadVariableOp�3transient_classifier/dense_c2/MatMul/ReadVariableOp�5transient_classifier/dense_me1/BiasAdd/ReadVariableOp�4transient_classifier/dense_me1/MatMul/ReadVariableOp�5transient_classifier/dense_me2/BiasAdd/ReadVariableOp�4transient_classifier/dense_me2/MatMul/ReadVariableOp�2transient_classifier/output/BiasAdd/ReadVariableOp�1transient_classifier/output/MatMul/ReadVariableOp�
;transient_classifier/data_augmentation/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
Etransient_classifier/data_augmentation/resizing/resize/ResizeBilinearResizeBilinearimage_inputDtransient_classifier/data_augmentation/resizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(�
1transient_classifier/conv_0/Conv2D/ReadVariableOpReadVariableOp:transient_classifier_conv_0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
"transient_classifier/conv_0/Conv2DConv2DVtransient_classifier/data_augmentation/resizing/resize/ResizeBilinear:resized_images:09transient_classifier/conv_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@*
paddingVALID*
strides
�
2transient_classifier/conv_0/BiasAdd/ReadVariableOpReadVariableOp;transient_classifier_conv_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#transient_classifier/conv_0/BiasAddBiasAdd+transient_classifier/conv_0/Conv2D:output:0:transient_classifier/conv_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@�
 transient_classifier/conv_0/ReluRelu,transient_classifier/conv_0/BiasAdd:output:0*
T0*/
_output_shapes
:���������::@�
#transient_classifier/pool_0/MaxPoolMaxPool.transient_classifier/conv_0/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
1transient_classifier/conv_1/Conv2D/ReadVariableOpReadVariableOp:transient_classifier_conv_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
"transient_classifier/conv_1/Conv2DConv2D,transient_classifier/pool_0/MaxPool:output:09transient_classifier/conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
2transient_classifier/conv_1/BiasAdd/ReadVariableOpReadVariableOp;transient_classifier_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#transient_classifier/conv_1/BiasAddBiasAdd+transient_classifier/conv_1/Conv2D:output:0:transient_classifier/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
 transient_classifier/conv_1/ReluRelu,transient_classifier/conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
#transient_classifier/pool_1/MaxPoolMaxPool.transient_classifier/conv_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
1transient_classifier/conv_2/Conv2D/ReadVariableOpReadVariableOp:transient_classifier_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
"transient_classifier/conv_2/Conv2DConv2D,transient_classifier/pool_1/MaxPool:output:09transient_classifier/conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
2transient_classifier/conv_2/BiasAdd/ReadVariableOpReadVariableOp;transient_classifier_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#transient_classifier/conv_2/BiasAddBiasAdd+transient_classifier/conv_2/Conv2D:output:0:transient_classifier/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
 transient_classifier/conv_2/ReluRelu,transient_classifier/conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
#transient_classifier/pool_2/MaxPoolMaxPool.transient_classifier/conv_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
s
"transient_classifier/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
$transient_classifier/flatten/ReshapeReshape,transient_classifier/pool_2/MaxPool:output:0+transient_classifier/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
*transient_classifier/feature_ranking/mul/yConst*
_output_shapes
:*
dtype0*U
valueLBJ"@ݖ�=�{�=��5=��=S<=�	5=p�9={�j=�Ѡ>���= J>�D�=s{�=��=z�h=|v�=�
(transient_classifier/feature_ranking/mulMul
meta_input3transient_classifier/feature_ranking/mul/y:output:0*
T0*'
_output_shapes
:����������
4transient_classifier/dense_me1/MatMul/ReadVariableOpReadVariableOp=transient_classifier_dense_me1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%transient_classifier/dense_me1/MatMulMatMul,transient_classifier/feature_ranking/mul:z:0<transient_classifier/dense_me1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5transient_classifier/dense_me1/BiasAdd/ReadVariableOpReadVariableOp>transient_classifier_dense_me1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&transient_classifier/dense_me1/BiasAddBiasAdd/transient_classifier/dense_me1/MatMul:product:0=transient_classifier/dense_me1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#transient_classifier/dense_me1/ReluRelu/transient_classifier/dense_me1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4transient_classifier/dense_me2/MatMul/ReadVariableOpReadVariableOp=transient_classifier_dense_me2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
%transient_classifier/dense_me2/MatMulMatMul1transient_classifier/dense_me1/Relu:activations:0<transient_classifier/dense_me2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5transient_classifier/dense_me2/BiasAdd/ReadVariableOpReadVariableOp>transient_classifier_dense_me2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&transient_classifier/dense_me2/BiasAddBiasAdd/transient_classifier/dense_me2/MatMul:product:0=transient_classifier/dense_me2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#transient_classifier/dense_me2/ReluRelu/transient_classifier/dense_me2/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
,transient_classifier/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
'transient_classifier/concatenate/concatConcatV2-transient_classifier/flatten/Reshape:output:01transient_classifier/dense_me2/Relu:activations:05transient_classifier/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
3transient_classifier/dense_c1/MatMul/ReadVariableOpReadVariableOp<transient_classifier_dense_c1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
$transient_classifier/dense_c1/MatMulMatMul0transient_classifier/concatenate/concat:output:0;transient_classifier/dense_c1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4transient_classifier/dense_c1/BiasAdd/ReadVariableOpReadVariableOp=transient_classifier_dense_c1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%transient_classifier/dense_c1/BiasAddBiasAdd.transient_classifier/dense_c1/MatMul:product:0<transient_classifier/dense_c1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"transient_classifier/dense_c1/ReluRelu.transient_classifier/dense_c1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
3transient_classifier/dense_c2/MatMul/ReadVariableOpReadVariableOp<transient_classifier_dense_c2_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
$transient_classifier/dense_c2/MatMulMatMul0transient_classifier/dense_c1/Relu:activations:0;transient_classifier/dense_c2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
4transient_classifier/dense_c2/BiasAdd/ReadVariableOpReadVariableOp=transient_classifier_dense_c2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
%transient_classifier/dense_c2/BiasAddBiasAdd.transient_classifier/dense_c2/MatMul:product:0<transient_classifier/dense_c2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
"transient_classifier/dense_c2/ReluRelu.transient_classifier/dense_c2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1transient_classifier/output/MatMul/ReadVariableOpReadVariableOp:transient_classifier_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
"transient_classifier/output/MatMulMatMul0transient_classifier/dense_c2/Relu:activations:09transient_classifier/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2transient_classifier/output/BiasAdd/ReadVariableOpReadVariableOp;transient_classifier_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#transient_classifier/output/BiasAddBiasAdd,transient_classifier/output/MatMul:product:0:transient_classifier/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#transient_classifier/output/SoftmaxSoftmax,transient_classifier/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������|
IdentityIdentity-transient_classifier/output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^transient_classifier/conv_0/BiasAdd/ReadVariableOp2^transient_classifier/conv_0/Conv2D/ReadVariableOp3^transient_classifier/conv_1/BiasAdd/ReadVariableOp2^transient_classifier/conv_1/Conv2D/ReadVariableOp3^transient_classifier/conv_2/BiasAdd/ReadVariableOp2^transient_classifier/conv_2/Conv2D/ReadVariableOp5^transient_classifier/dense_c1/BiasAdd/ReadVariableOp4^transient_classifier/dense_c1/MatMul/ReadVariableOp5^transient_classifier/dense_c2/BiasAdd/ReadVariableOp4^transient_classifier/dense_c2/MatMul/ReadVariableOp6^transient_classifier/dense_me1/BiasAdd/ReadVariableOp5^transient_classifier/dense_me1/MatMul/ReadVariableOp6^transient_classifier/dense_me2/BiasAdd/ReadVariableOp5^transient_classifier/dense_me2/MatMul/ReadVariableOp3^transient_classifier/output/BiasAdd/ReadVariableOp2^transient_classifier/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 2h
2transient_classifier/conv_0/BiasAdd/ReadVariableOp2transient_classifier/conv_0/BiasAdd/ReadVariableOp2f
1transient_classifier/conv_0/Conv2D/ReadVariableOp1transient_classifier/conv_0/Conv2D/ReadVariableOp2h
2transient_classifier/conv_1/BiasAdd/ReadVariableOp2transient_classifier/conv_1/BiasAdd/ReadVariableOp2f
1transient_classifier/conv_1/Conv2D/ReadVariableOp1transient_classifier/conv_1/Conv2D/ReadVariableOp2h
2transient_classifier/conv_2/BiasAdd/ReadVariableOp2transient_classifier/conv_2/BiasAdd/ReadVariableOp2f
1transient_classifier/conv_2/Conv2D/ReadVariableOp1transient_classifier/conv_2/Conv2D/ReadVariableOp2l
4transient_classifier/dense_c1/BiasAdd/ReadVariableOp4transient_classifier/dense_c1/BiasAdd/ReadVariableOp2j
3transient_classifier/dense_c1/MatMul/ReadVariableOp3transient_classifier/dense_c1/MatMul/ReadVariableOp2l
4transient_classifier/dense_c2/BiasAdd/ReadVariableOp4transient_classifier/dense_c2/BiasAdd/ReadVariableOp2j
3transient_classifier/dense_c2/MatMul/ReadVariableOp3transient_classifier/dense_c2/MatMul/ReadVariableOp2n
5transient_classifier/dense_me1/BiasAdd/ReadVariableOp5transient_classifier/dense_me1/BiasAdd/ReadVariableOp2l
4transient_classifier/dense_me1/MatMul/ReadVariableOp4transient_classifier/dense_me1/MatMul/ReadVariableOp2n
5transient_classifier/dense_me2/BiasAdd/ReadVariableOp5transient_classifier/dense_me2/BiasAdd/ReadVariableOp2l
4transient_classifier/dense_me2/MatMul/ReadVariableOp4transient_classifier/dense_me2/MatMul/ReadVariableOp2h
2transient_classifier/output/BiasAdd/ReadVariableOp2transient_classifier/output/BiasAdd/ReadVariableOp2f
1transient_classifier/output/MatMul/ReadVariableOp1transient_classifier/output/MatMul/ReadVariableOp:SO
'
_output_shapes
:���������
$
_user_specified_name
meta_input:\ X
/
_output_shapes
:���������<<
%
_user_specified_nameimage_input
�
g
K__inference_feature_ranking_layer_call_and_return_conditional_losses_616368

inputs
identity�
mul/yConst*
_output_shapes
:*
dtype0*U
valueLBJ"@ݖ�=�{�=��5=��=S<=�	5=p�9={�j=�Ѡ>���= J>�D�=s{�=��=z�h=|v�=T
mulMulinputsmul/y:output:0*
T0*'
_output_shapes
:���������O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_me1_layer_call_and_return_conditional_losses_616381

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
B__inference_output_layer_call_and_return_conditional_losses_616458

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
E__inference_dense_me1_layer_call_and_return_conditional_losses_617684

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5__inference_transient_classifier_layer_call_fn_616622
image_input

meta_input
unknown:	
	unknown_0:	#
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	� 

unknown_14: 

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimage_input
meta_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������<<:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:SO
'
_output_shapes
:���������
$
_user_specified_name
meta_input:\ X
/
_output_shapes
:���������<<
%
_user_specified_nameimage_input
��
�
M__inference_data_augmentation_layer_call_and_return_conditional_losses_616294

inputsK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	
identity��4random_flip/stateful_uniform_full_int/RngReadAndSkip�6random_flip/stateful_uniform_full_int_1/RngReadAndSkip�/random_rotation/stateful_uniform/RngReadAndSkipe
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(u
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:u
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: n
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:�
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0k
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :�
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	`
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R �
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
?random_flip/stateless_random_flip_left_right/control_dependencyIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*1
_class'
%#loc:@resizing/resize/ResizeBilinear*/
_output_shapes
:���������<<�
2random_flip/stateless_random_flip_left_right/ShapeShapeHrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::���
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:�
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0* 
_output_shapes
::�
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:���������~
<random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
:random_flip/stateless_random_flip_left_right/Reshape/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
4random_flip/stateless_random_flip_left_right/ReshapeReshapeIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Crandom_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
2random_flip/stateless_random_flip_left_right/RoundRound=random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:����������
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0Drandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:���������<<�
0random_flip/stateless_random_flip_left_right/mulMul6random_flip/stateless_random_flip_left_right/Round:y:0?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:���������<<w
2random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
0random_flip/stateless_random_flip_left_right/subSub;random_flip/stateless_random_flip_left_right/sub/x:output:06random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:����������
2random_flip/stateless_random_flip_left_right/mul_1Mul4random_flip/stateless_random_flip_left_right/sub:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:���������<<�
0random_flip/stateless_random_flip_left_right/addAddV24random_flip/stateless_random_flip_left_right/mul:z:06random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:���������<<w
-random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
-random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,random_flip/stateful_uniform_full_int_1/ProdProd6random_flip/stateful_uniform_full_int_1/shape:output:06random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: p
.random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
.random_flip/stateful_uniform_full_int_1/Cast_1Cast5random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource7random_flip/stateful_uniform_full_int_1/Cast/x:output:02random_flip/stateful_uniform_full_int_1/Cast_1:y:05^random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:�
;random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
/random_flip/stateful_uniform_full_int_1/BitcastBitcast>random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Frandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
1random_flip/stateful_uniform_full_int_1/Bitcast_1Bitcast@random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0m
+random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :�
'random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV26random_flip/stateful_uniform_full_int_1/shape:output:0:random_flip/stateful_uniform_full_int_1/Bitcast_1:output:08random_flip/stateful_uniform_full_int_1/Bitcast:output:04random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	b
random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R �
random_flip/stack_1Pack0random_flip/stateful_uniform_full_int_1:output:0!random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:r
!random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
random_flip/strided_slice_1StridedSlicerandom_flip/stack_1:output:0*random_flip/strided_slice_1/stack:output:0,random_flip/strided_slice_1/stack_1:output:0,random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
<random_flip/stateless_random_flip_up_down/control_dependencyIdentity4random_flip/stateless_random_flip_left_right/add:z:0*
T0*C
_class9
75loc:@random_flip/stateless_random_flip_left_right/add*/
_output_shapes
:���������<<�
/random_flip/stateless_random_flip_up_down/ShapeShapeErandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::���
=random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7random_flip/stateless_random_flip_up_down/strided_sliceStridedSlice8random_flip/stateless_random_flip_up_down/Shape:output:0Frandom_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:�
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip/strided_slice_1:output:0* 
_output_shapes
::�
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Qrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0erandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0irandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubOrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMuldrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:���������{
9random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :{
9random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :{
9random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
7random_flip/stateless_random_flip_up_down/Reshape/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
1random_flip/stateless_random_flip_up_down/ReshapeReshapeFrandom_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0@random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
/random_flip/stateless_random_flip_up_down/RoundRound:random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:����������
8random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
3random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0Arandom_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*/
_output_shapes
:���������<<�
-random_flip/stateless_random_flip_up_down/mulMul3random_flip/stateless_random_flip_up_down/Round:y:0<random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*/
_output_shapes
:���������<<t
/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-random_flip/stateless_random_flip_up_down/subSub8random_flip/stateless_random_flip_up_down/sub/x:output:03random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:����������
/random_flip/stateless_random_flip_up_down/mul_1Mul1random_flip/stateless_random_flip_up_down/sub:z:0Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*/
_output_shapes
:���������<<�
-random_flip/stateless_random_flip_up_down/addAddV21random_flip/stateless_random_flip_up_down/mul:z:03random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*/
_output_shapes
:���������<<�
random_rotation/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
::��m
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������z
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������q
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: x
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������z
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������q
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: �
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:i
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *���i
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��@p
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: i
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:~
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: �
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:����������
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������j
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ~
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: �
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������~
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������n
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:����������
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������p
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::��}
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_maskp
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:p
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������m
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
random_rotation/transform/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
::��w
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:i
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV31random_flip/stateless_random_flip_up_down/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*/
_output_shapes
:���������<<*
dtype0*
	fill_mode	NEAREST*
interpolation
BILINEAR�
IdentityIdentityIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*/
_output_shapes
:���������<<�
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip7^random_flip/stateful_uniform_full_int_1/RngReadAndSkip0^random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<: : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip/stateful_uniform_full_int_1/RngReadAndSkip6random_flip/stateful_uniform_full_int_1/RngReadAndSkip2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
X
,__inference_concatenate_layer_call_fn_617710
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_616411a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
*__inference_dense_me1_layer_call_fn_617673

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_me1_layer_call_and_return_conditional_losses_616381p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617395
inputs_image_input
inputs_meta_input?
%conv_0_conv2d_readvariableop_resource:@4
&conv_0_biasadd_readvariableop_resource:@@
%conv_1_conv2d_readvariableop_resource:@�5
&conv_1_biasadd_readvariableop_resource:	�A
%conv_2_conv2d_readvariableop_resource:��5
&conv_2_biasadd_readvariableop_resource:	�;
(dense_me1_matmul_readvariableop_resource:	�8
)dense_me1_biasadd_readvariableop_resource:	�<
(dense_me2_matmul_readvariableop_resource:
��8
)dense_me2_biasadd_readvariableop_resource:	�;
'dense_c1_matmul_readvariableop_resource:
��7
(dense_c1_biasadd_readvariableop_resource:	�:
'dense_c2_matmul_readvariableop_resource:	� 6
(dense_c2_biasadd_readvariableop_resource: 7
%output_matmul_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identity��conv_0/BiasAdd/ReadVariableOp�conv_0/Conv2D/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�dense_c1/BiasAdd/ReadVariableOp�dense_c1/MatMul/ReadVariableOp�dense_c2/BiasAdd/ReadVariableOp�dense_c2/MatMul/ReadVariableOp� dense_me1/BiasAdd/ReadVariableOp�dense_me1/MatMul/ReadVariableOp� dense_me2/BiasAdd/ReadVariableOp�dense_me2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpw
&data_augmentation/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
0data_augmentation/resizing/resize/ResizeBilinearResizeBilinearinputs_image_input/data_augmentation/resizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(�
conv_0/Conv2D/ReadVariableOpReadVariableOp%conv_0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv_0/Conv2DConv2DAdata_augmentation/resizing/resize/ResizeBilinear:resized_images:0$conv_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@*
paddingVALID*
strides
�
conv_0/BiasAdd/ReadVariableOpReadVariableOp&conv_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv_0/BiasAddBiasAddconv_0/Conv2D:output:0%conv_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������::@f
conv_0/ReluReluconv_0/BiasAdd:output:0*
T0*/
_output_shapes
:���������::@�
pool_0/MaxPoolMaxPoolconv_0/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv_1/Conv2DConv2Dpool_0/MaxPool:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
pool_1/MaxPoolMaxPoolconv_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv_2/Conv2DConv2Dpool_1/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
pool_2/MaxPoolMaxPoolconv_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ~
flatten/ReshapeReshapepool_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
feature_ranking/mul/yConst*
_output_shapes
:*
dtype0*U
valueLBJ"@ݖ�=�{�=��5=��=S<=�	5=p�9={�j=�Ѡ>���= J>�D�=s{�=��=z�h=|v�=
feature_ranking/mulMulinputs_meta_inputfeature_ranking/mul/y:output:0*
T0*'
_output_shapes
:����������
dense_me1/MatMul/ReadVariableOpReadVariableOp(dense_me1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_me1/MatMulMatMulfeature_ranking/mul:z:0'dense_me1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_me1/BiasAdd/ReadVariableOpReadVariableOp)dense_me1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_me1/BiasAddBiasAdddense_me1/MatMul:product:0(dense_me1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_me1/ReluReludense_me1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_me2/MatMul/ReadVariableOpReadVariableOp(dense_me2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_me2/MatMulMatMuldense_me1/Relu:activations:0'dense_me2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_me2/BiasAdd/ReadVariableOpReadVariableOp)dense_me2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_me2/BiasAddBiasAdddense_me2/MatMul:product:0(dense_me2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_me2/ReluReludense_me2/BiasAdd:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0dense_me2/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_c1/MatMul/ReadVariableOpReadVariableOp'dense_c1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_c1/MatMulMatMulconcatenate/concat:output:0&dense_c1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_c1/BiasAdd/ReadVariableOpReadVariableOp(dense_c1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_c1/BiasAddBiasAdddense_c1/MatMul:product:0'dense_c1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_c1/ReluReludense_c1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_c2/MatMul/ReadVariableOpReadVariableOp'dense_c2_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_c2/MatMulMatMuldense_c1/Relu:activations:0&dense_c2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_c2/BiasAdd/ReadVariableOpReadVariableOp(dense_c2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_c2/BiasAddBiasAdddense_c2/MatMul:product:0'dense_c2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_c2/ReluReludense_c2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output/MatMulMatMuldense_c2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������g
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv_0/BiasAdd/ReadVariableOp^conv_0/Conv2D/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp ^dense_c1/BiasAdd/ReadVariableOp^dense_c1/MatMul/ReadVariableOp ^dense_c2/BiasAdd/ReadVariableOp^dense_c2/MatMul/ReadVariableOp!^dense_me1/BiasAdd/ReadVariableOp ^dense_me1/MatMul/ReadVariableOp!^dense_me2/BiasAdd/ReadVariableOp ^dense_me2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������<<:���������: : : : : : : : : : : : : : : : 2>
conv_0/BiasAdd/ReadVariableOpconv_0/BiasAdd/ReadVariableOp2<
conv_0/Conv2D/ReadVariableOpconv_0/Conv2D/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2B
dense_c1/BiasAdd/ReadVariableOpdense_c1/BiasAdd/ReadVariableOp2@
dense_c1/MatMul/ReadVariableOpdense_c1/MatMul/ReadVariableOp2B
dense_c2/BiasAdd/ReadVariableOpdense_c2/BiasAdd/ReadVariableOp2@
dense_c2/MatMul/ReadVariableOpdense_c2/MatMul/ReadVariableOp2D
 dense_me1/BiasAdd/ReadVariableOp dense_me1/BiasAdd/ReadVariableOp2B
dense_me1/MatMul/ReadVariableOpdense_me1/MatMul/ReadVariableOp2D
 dense_me2/BiasAdd/ReadVariableOp dense_me2/BiasAdd/ReadVariableOp2B
dense_me2/MatMul/ReadVariableOpdense_me2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_meta_input:c _
/
_output_shapes
:���������<<
,
_user_specified_nameinputs_image_input
��
�
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617636

inputsK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	
identity��4random_flip/stateful_uniform_full_int/RngReadAndSkip�6random_flip/stateful_uniform_full_int_1/RngReadAndSkip�/random_rotation/stateful_uniform/RngReadAndSkipe
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"<   <   �
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*/
_output_shapes
:���������<<*
half_pixel_centers(u
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:u
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: n
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:�
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0k
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :�
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	`
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R �
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
?random_flip/stateless_random_flip_left_right/control_dependencyIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*1
_class'
%#loc:@resizing/resize/ResizeBilinear*/
_output_shapes
:���������<<�
2random_flip/stateless_random_flip_left_right/ShapeShapeHrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::���
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:�
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0* 
_output_shapes
::�
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:���������~
<random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
:random_flip/stateless_random_flip_left_right/Reshape/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
4random_flip/stateless_random_flip_left_right/ReshapeReshapeIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Crandom_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
2random_flip/stateless_random_flip_left_right/RoundRound=random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:����������
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0Drandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:���������<<�
0random_flip/stateless_random_flip_left_right/mulMul6random_flip/stateless_random_flip_left_right/Round:y:0?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:���������<<w
2random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
0random_flip/stateless_random_flip_left_right/subSub;random_flip/stateless_random_flip_left_right/sub/x:output:06random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:����������
2random_flip/stateless_random_flip_left_right/mul_1Mul4random_flip/stateless_random_flip_left_right/sub:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:���������<<�
0random_flip/stateless_random_flip_left_right/addAddV24random_flip/stateless_random_flip_left_right/mul:z:06random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:���������<<w
-random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
-random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,random_flip/stateful_uniform_full_int_1/ProdProd6random_flip/stateful_uniform_full_int_1/shape:output:06random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: p
.random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
.random_flip/stateful_uniform_full_int_1/Cast_1Cast5random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource7random_flip/stateful_uniform_full_int_1/Cast/x:output:02random_flip/stateful_uniform_full_int_1/Cast_1:y:05^random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:�
;random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
/random_flip/stateful_uniform_full_int_1/BitcastBitcast>random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Frandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
1random_flip/stateful_uniform_full_int_1/Bitcast_1Bitcast@random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0m
+random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :�
'random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV26random_flip/stateful_uniform_full_int_1/shape:output:0:random_flip/stateful_uniform_full_int_1/Bitcast_1:output:08random_flip/stateful_uniform_full_int_1/Bitcast:output:04random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	b
random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R �
random_flip/stack_1Pack0random_flip/stateful_uniform_full_int_1:output:0!random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:r
!random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
random_flip/strided_slice_1StridedSlicerandom_flip/stack_1:output:0*random_flip/strided_slice_1/stack:output:0,random_flip/strided_slice_1/stack_1:output:0,random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
<random_flip/stateless_random_flip_up_down/control_dependencyIdentity4random_flip/stateless_random_flip_left_right/add:z:0*
T0*C
_class9
75loc:@random_flip/stateless_random_flip_left_right/add*/
_output_shapes
:���������<<�
/random_flip/stateless_random_flip_up_down/ShapeShapeErandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::���
=random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7random_flip/stateless_random_flip_up_down/strided_sliceStridedSlice8random_flip/stateless_random_flip_up_down/Shape:output:0Frandom_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:�
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip/strided_slice_1:output:0* 
_output_shapes
::�
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Qrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0erandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0irandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubOrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMuldrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:���������{
9random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :{
9random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :{
9random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
7random_flip/stateless_random_flip_up_down/Reshape/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
1random_flip/stateless_random_flip_up_down/ReshapeReshapeFrandom_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0@random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
/random_flip/stateless_random_flip_up_down/RoundRound:random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:����������
8random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
3random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0Arandom_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*/
_output_shapes
:���������<<�
-random_flip/stateless_random_flip_up_down/mulMul3random_flip/stateless_random_flip_up_down/Round:y:0<random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*/
_output_shapes
:���������<<t
/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-random_flip/stateless_random_flip_up_down/subSub8random_flip/stateless_random_flip_up_down/sub/x:output:03random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:����������
/random_flip/stateless_random_flip_up_down/mul_1Mul1random_flip/stateless_random_flip_up_down/sub:z:0Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*/
_output_shapes
:���������<<�
-random_flip/stateless_random_flip_up_down/addAddV21random_flip/stateless_random_flip_up_down/mul:z:03random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*/
_output_shapes
:���������<<�
random_rotation/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
::��m
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������z
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������q
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: x
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������z
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������q
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: �
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:i
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *���i
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��@p
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: i
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:~
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0�
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: �
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:����������
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������j
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ~
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: �
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������~
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������n
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������l
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: �
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:����������
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������p
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:����������
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::��}
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:����������
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_maskp
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:p
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������m
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
random_rotation/transform/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
::��w
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:i
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV31random_flip/stateless_random_flip_up_down/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*/
_output_shapes
:���������<<*
dtype0*
	fill_mode	NEAREST*
interpolation
BILINEAR�
IdentityIdentityIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*/
_output_shapes
:���������<<�
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip7^random_flip/stateful_uniform_full_int_1/RngReadAndSkip0^random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<: : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip/stateful_uniform_full_int_1/RngReadAndSkip6random_flip/stateful_uniform_full_int_1/RngReadAndSkip2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_617653

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_conv_0_layer_call_fn_617786

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������::@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_616311w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������::@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<<
 
_user_specified_nameinputs
�

�
B__inference_output_layer_call_and_return_conditional_losses_617777

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_conv_2_layer_call_fn_617846

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_616347x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
image_input<
serving_default_image_input:0���������<<
A

meta_input3
serving_default_meta_input:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
neurons
	
label_dict


cnn_layers
feature_importance
data_augmentation
flatten
meta_weighted
dense_m1
dense_m2
concatenate
dense_c1
dense_c2
output_layer
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
-trace_0
.trace_1
/trace_2
0trace_32�
5__inference_transient_classifier_layer_call_fn_616622
5__inference_transient_classifier_layer_call_fn_616712
5__inference_transient_classifier_layer_call_fn_616992
5__inference_transient_classifier_layer_call_fn_617030�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z-trace_0z.trace_1z/trace_2z0trace_3
�
1trace_0
2trace_1
3trace_2
4trace_32�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616465
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616523
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617323
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617395�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z1trace_0z2trace_1z3trace_2z4trace_3
�B�
!__inference__wrapped_model_616026image_input
meta_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
50
61
72"
trackable_list_wrapper
 "
trackable_dict_wrapper
5
80
91
:2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
AResizing
B
RandomFlip
CRandomRotation"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
titer

ubeta_1

vbeta_2
	wdecaym�m�m�m�m�m�m�m� m�!m�"m�#m�$m�%m�&m�'m�v�v�v�v�v�v�v�v� v�!v�"v�#v�$v�%v�&v�'v�"
	optimizer
,
xserving_default"
signature_map
':%@2conv_0/kernel
:@2conv_0/bias
(:&@�2conv_1/kernel
:�2conv_1/bias
):'��2conv_2/kernel
:�2conv_2/bias
$:"2feature_ranking/kernel
#:!	�2dense_me1/kernel
:�2dense_me1/bias
$:"
��2dense_me2/kernel
:�2dense_me2/bias
#:!
��2dense_c1/kernel
:�2dense_c1/bias
": 	� 2dense_c2/kernel
: 2dense_c2/bias
: 2output/kernel
:2output/bias
 "
trackable_list_wrapper
�
y0
z1
{2
|3
}4
~5
6
7
8
9
10
11
12
13
14"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_transient_classifier_layer_call_fn_616622image_input
meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
5__inference_transient_classifier_layer_call_fn_616712image_input
meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
5__inference_transient_classifier_layer_call_fn_616992inputs_image_inputinputs_meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
5__inference_transient_classifier_layer_call_fn_617030inputs_image_inputinputs_meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616465image_input
meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616523image_input
meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617323inputs_image_inputinputs_meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617395inputs_image_inputinputs_meta_input"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_data_augmentation_layer_call_fn_617404
2__inference_data_augmentation_layer_call_fn_617409�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617636
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617642�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�factor"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_617647�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_617653�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_feature_ranking_layer_call_fn_617658�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_feature_ranking_layer_call_and_return_conditional_losses_617664�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_me1_layer_call_fn_617673�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_me1_layer_call_and_return_conditional_losses_617684�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_me2_layer_call_fn_617693�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_me2_layer_call_and_return_conditional_losses_617704�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_layer_call_fn_617710�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_layer_call_and_return_conditional_losses_617717�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_c1_layer_call_fn_617726�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_c1_layer_call_and_return_conditional_losses_617737�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_c2_layer_call_fn_617746�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_c2_layer_call_and_return_conditional_losses_617757�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_output_layer_call_fn_617766�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_output_layer_call_and_return_conditional_losses_617777�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
�B�
$__inference_signature_wrapper_616950image_input
meta_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_data_augmentation_layer_call_fn_617404inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
2__inference_data_augmentation_layer_call_fn_617409inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617636inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617642inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/
�
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/
�
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_flatten_layer_call_fn_617647inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_617653inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_feature_ranking_layer_call_fn_617658inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_feature_ranking_layer_call_and_return_conditional_losses_617664inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_me1_layer_call_fn_617673inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_me1_layer_call_and_return_conditional_losses_617684inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_me2_layer_call_fn_617693inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_me2_layer_call_and_return_conditional_losses_617704inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_layer_call_fn_617710inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_layer_call_and_return_conditional_losses_617717inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_c1_layer_call_fn_617726inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_c1_layer_call_and_return_conditional_losses_617737inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_c2_layer_call_fn_617746inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_c2_layer_call_and_return_conditional_losses_617757inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_output_layer_call_fn_617766inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_output_layer_call_and_return_conditional_losses_617777inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv_0_layer_call_fn_617786�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv_0_layer_call_and_return_conditional_losses_617797�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_pool_0_layer_call_fn_617802�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_pool_0_layer_call_and_return_conditional_losses_617807�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv_1_layer_call_fn_617816�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv_1_layer_call_and_return_conditional_losses_617827�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_pool_1_layer_call_fn_617832�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_pool_1_layer_call_and_return_conditional_losses_617837�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv_2_layer_call_fn_617846�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv_2_layer_call_and_return_conditional_losses_617857�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_pool_2_layer_call_fn_617862�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_pool_2_layer_call_and_return_conditional_losses_617867�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
�
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
�
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv_0_layer_call_fn_617786inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv_0_layer_call_and_return_conditional_losses_617797inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_pool_0_layer_call_fn_617802inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_pool_0_layer_call_and_return_conditional_losses_617807inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv_1_layer_call_fn_617816inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv_1_layer_call_and_return_conditional_losses_617827inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_pool_1_layer_call_fn_617832inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_pool_1_layer_call_and_return_conditional_losses_617837inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv_2_layer_call_fn_617846inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv_2_layer_call_and_return_conditional_losses_617857inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_pool_2_layer_call_fn_617862inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_pool_2_layer_call_and_return_conditional_losses_617867inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2:0	2&data_augmentation/random_flip/StateVar
6:4	2*data_augmentation/random_rotation/StateVar
,:*@2Adam/conv_0/kernel/m
:@2Adam/conv_0/bias/m
-:+@�2Adam/conv_1/kernel/m
:�2Adam/conv_1/bias/m
.:,��2Adam/conv_2/kernel/m
:�2Adam/conv_2/bias/m
(:&	�2Adam/dense_me1/kernel/m
": �2Adam/dense_me1/bias/m
):'
��2Adam/dense_me2/kernel/m
": �2Adam/dense_me2/bias/m
(:&
��2Adam/dense_c1/kernel/m
!:�2Adam/dense_c1/bias/m
':%	� 2Adam/dense_c2/kernel/m
 : 2Adam/dense_c2/bias/m
$:" 2Adam/output/kernel/m
:2Adam/output/bias/m
,:*@2Adam/conv_0/kernel/v
:@2Adam/conv_0/bias/v
-:+@�2Adam/conv_1/kernel/v
:�2Adam/conv_1/bias/v
.:,��2Adam/conv_2/kernel/v
:�2Adam/conv_2/bias/v
(:&	�2Adam/dense_me1/kernel/v
": �2Adam/dense_me1/bias/v
):'
��2Adam/dense_me2/kernel/v
": �2Adam/dense_me2/bias/v
(:&
��2Adam/dense_c1/kernel/v
!:�2Adam/dense_c1/bias/v
':%	� 2Adam/dense_c2/kernel/v
 : 2Adam/dense_c2/bias/v
$:" 2Adam/output/kernel/v
:2Adam/output/bias/v�
!__inference__wrapped_model_616026� !"#$%&'���
z�w
u�r
<
image_input-�*
image_input���������<<
2

meta_input$�!

meta_input���������
� "3�0
.
output_1"�
output_1����������
G__inference_concatenate_layer_call_and_return_conditional_losses_617717�\�Y
R�O
M�J
#� 
inputs_0����������
#� 
inputs_1����������
� "-�*
#� 
tensor_0����������
� �
,__inference_concatenate_layer_call_fn_617710�\�Y
R�O
M�J
#� 
inputs_0����������
#� 
inputs_1����������
� ""�
unknown�����������
B__inference_conv_0_layer_call_and_return_conditional_losses_617797s7�4
-�*
(�%
inputs���������<<
� "4�1
*�'
tensor_0���������::@
� �
'__inference_conv_0_layer_call_fn_617786h7�4
-�*
(�%
inputs���������<<
� ")�&
unknown���������::@�
B__inference_conv_1_layer_call_and_return_conditional_losses_617827t7�4
-�*
(�%
inputs���������@
� "5�2
+�(
tensor_0����������
� �
'__inference_conv_1_layer_call_fn_617816i7�4
-�*
(�%
inputs���������@
� "*�'
unknown�����������
B__inference_conv_2_layer_call_and_return_conditional_losses_617857u8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
'__inference_conv_2_layer_call_fn_617846j8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617636���G�D
-�*
(�%
inputs���������<<
�

trainingp"4�1
*�'
tensor_0���������<<
� �
M__inference_data_augmentation_layer_call_and_return_conditional_losses_617642G�D
-�*
(�%
inputs���������<<
�

trainingp "4�1
*�'
tensor_0���������<<
� �
2__inference_data_augmentation_layer_call_fn_617404z��G�D
-�*
(�%
inputs���������<<
�

trainingp")�&
unknown���������<<�
2__inference_data_augmentation_layer_call_fn_617409tG�D
-�*
(�%
inputs���������<<
�

trainingp ")�&
unknown���������<<�
D__inference_dense_c1_layer_call_and_return_conditional_losses_617737e"#0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_c1_layer_call_fn_617726Z"#0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_c2_layer_call_and_return_conditional_losses_617757d$%0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_c2_layer_call_fn_617746Y$%0�-
&�#
!�
inputs����������
� "!�
unknown��������� �
E__inference_dense_me1_layer_call_and_return_conditional_losses_617684d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_me1_layer_call_fn_617673Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_me2_layer_call_and_return_conditional_losses_617704e !0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_me2_layer_call_fn_617693Z !0�-
&�#
!�
inputs����������
� ""�
unknown�����������
K__inference_feature_ranking_layer_call_and_return_conditional_losses_617664_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
0__inference_feature_ranking_layer_call_fn_617658T/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_flatten_layer_call_and_return_conditional_losses_617653i8�5
.�+
)�&
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_flatten_layer_call_fn_617647^8�5
.�+
)�&
inputs����������
� ""�
unknown�����������
B__inference_output_layer_call_and_return_conditional_losses_617777c&'/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
'__inference_output_layer_call_fn_617766X&'/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
B__inference_pool_0_layer_call_and_return_conditional_losses_617807�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
'__inference_pool_0_layer_call_fn_617802�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
B__inference_pool_1_layer_call_and_return_conditional_losses_617837�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
'__inference_pool_1_layer_call_fn_617832�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
B__inference_pool_2_layer_call_and_return_conditional_losses_617867�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
'__inference_pool_2_layer_call_fn_617862�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
$__inference_signature_wrapper_616950� !"#$%&'�|
� 
u�r
<
image_input-�*
image_input���������<<
2

meta_input$�!

meta_input���������"3�0
.
output_1"�
output_1����������
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616465��� !"#$%&'���
z�w
u�r
<
image_input-�*
image_input���������<<
2

meta_input$�!

meta_input���������
�

trainingp",�)
"�
tensor_0���������
� �
P__inference_transient_classifier_layer_call_and_return_conditional_losses_616523� !"#$%&'���
z�w
u�r
<
image_input-�*
image_input���������<<
2

meta_input$�!

meta_input���������
�

trainingp ",�)
"�
tensor_0���������
� �
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617323��� !"#$%&'���
���
���
C
image_input4�1
inputs_image_input���������<<
9

meta_input+�(
inputs_meta_input���������
�

trainingp",�)
"�
tensor_0���������
� �
P__inference_transient_classifier_layer_call_and_return_conditional_losses_617395� !"#$%&'���
���
���
C
image_input4�1
inputs_image_input���������<<
9

meta_input+�(
inputs_meta_input���������
�

trainingp ",�)
"�
tensor_0���������
� �
5__inference_transient_classifier_layer_call_fn_616622��� !"#$%&'���
z�w
u�r
<
image_input-�*
image_input���������<<
2

meta_input$�!

meta_input���������
�

trainingp"!�
unknown����������
5__inference_transient_classifier_layer_call_fn_616712� !"#$%&'���
z�w
u�r
<
image_input-�*
image_input���������<<
2

meta_input$�!

meta_input���������
�

trainingp "!�
unknown����������
5__inference_transient_classifier_layer_call_fn_616992��� !"#$%&'���
���
���
C
image_input4�1
inputs_image_input���������<<
9

meta_input+�(
inputs_meta_input���������
�

trainingp"!�
unknown����������
5__inference_transient_classifier_layer_call_fn_617030� !"#$%&'���
���
���
C
image_input4�1
inputs_image_input���������<<
9

meta_input+�(
inputs_meta_input���������
�

trainingp "!�
unknown���������