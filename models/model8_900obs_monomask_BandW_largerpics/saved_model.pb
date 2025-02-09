��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
executor_typestring �
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
 �"serve*2.4.12v2.4.1-0-g85c8b2a817f8��	
�
conv2d_10803/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv2d_10803/kernel
�
'conv2d_10803/kernel/Read/ReadVariableOpReadVariableOpconv2d_10803/kernel*&
_output_shapes
: *
dtype0
z
conv2d_10803/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_10803/bias
s
%conv2d_10803/bias/Read/ReadVariableOpReadVariableOpconv2d_10803/bias*
_output_shapes
: *
dtype0
�
conv2d_10804/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameconv2d_10804/kernel
�
'conv2d_10804/kernel/Read/ReadVariableOpReadVariableOpconv2d_10804/kernel*&
_output_shapes
: @*
dtype0
z
conv2d_10804/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_10804/bias
s
%conv2d_10804/bias/Read/ReadVariableOpReadVariableOpconv2d_10804/bias*
_output_shapes
:@*
dtype0
�
conv2d_10805/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*$
shared_nameconv2d_10805/kernel
�
'conv2d_10805/kernel/Read/ReadVariableOpReadVariableOpconv2d_10805/kernel*'
_output_shapes
:@�*
dtype0
{
conv2d_10805/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameconv2d_10805/bias
t
%conv2d_10805/bias/Read/ReadVariableOpReadVariableOpconv2d_10805/bias*
_output_shapes	
:�*
dtype0

dense_6303/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�dx*"
shared_namedense_6303/kernel
x
%dense_6303/kernel/Read/ReadVariableOpReadVariableOpdense_6303/kernel*
_output_shapes
:	�dx*
dtype0
v
dense_6303/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x* 
shared_namedense_6303/bias
o
#dense_6303/bias/Read/ReadVariableOpReadVariableOpdense_6303/bias*
_output_shapes
:x*
dtype0
~
dense_6304/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*"
shared_namedense_6304/kernel
w
%dense_6304/kernel/Read/ReadVariableOpReadVariableOpdense_6304/kernel*
_output_shapes

:x<*
dtype0
v
dense_6304/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<* 
shared_namedense_6304/bias
o
#dense_6304/bias/Read/ReadVariableOpReadVariableOpdense_6304/bias*
_output_shapes
:<*
dtype0
~
dense_6305/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*"
shared_namedense_6305/kernel
w
%dense_6305/kernel/Read/ReadVariableOpReadVariableOpdense_6305/kernel*
_output_shapes

:<*
dtype0
v
dense_6305/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_6305/bias
o
#dense_6305/bias/Read/ReadVariableOpReadVariableOpdense_6305/bias*
_output_shapes
:*
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/conv2d_10803/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv2d_10803/kernel/m
�
.Adam/conv2d_10803/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10803/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_10803/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_10803/bias/m
�
,Adam/conv2d_10803/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10803/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_10804/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv2d_10804/kernel/m
�
.Adam/conv2d_10804/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10804/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/conv2d_10804/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_10804/bias/m
�
,Adam/conv2d_10804/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10804/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_10805/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameAdam/conv2d_10805/kernel/m
�
.Adam/conv2d_10805/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10805/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/conv2d_10805/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv2d_10805/bias/m
�
,Adam/conv2d_10805/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10805/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_6303/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�dx*)
shared_nameAdam/dense_6303/kernel/m
�
,Adam/dense_6303/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6303/kernel/m*
_output_shapes
:	�dx*
dtype0
�
Adam/dense_6303/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*'
shared_nameAdam/dense_6303/bias/m
}
*Adam/dense_6303/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6303/bias/m*
_output_shapes
:x*
dtype0
�
Adam/dense_6304/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*)
shared_nameAdam/dense_6304/kernel/m
�
,Adam/dense_6304/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6304/kernel/m*
_output_shapes

:x<*
dtype0
�
Adam/dense_6304/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_6304/bias/m
}
*Adam/dense_6304/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6304/bias/m*
_output_shapes
:<*
dtype0
�
Adam/dense_6305/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameAdam/dense_6305/kernel/m
�
,Adam/dense_6305/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6305/kernel/m*
_output_shapes

:<*
dtype0
�
Adam/dense_6305/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_6305/bias/m
}
*Adam/dense_6305/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6305/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_10803/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv2d_10803/kernel/v
�
.Adam/conv2d_10803/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10803/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_10803/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_10803/bias/v
�
,Adam/conv2d_10803/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10803/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_10804/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv2d_10804/kernel/v
�
.Adam/conv2d_10804/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10804/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/conv2d_10804/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_10804/bias/v
�
,Adam/conv2d_10804/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10804/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_10805/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameAdam/conv2d_10805/kernel/v
�
.Adam/conv2d_10805/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10805/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/conv2d_10805/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv2d_10805/bias/v
�
,Adam/conv2d_10805/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10805/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_6303/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�dx*)
shared_nameAdam/dense_6303/kernel/v
�
,Adam/dense_6303/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6303/kernel/v*
_output_shapes
:	�dx*
dtype0
�
Adam/dense_6303/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*'
shared_nameAdam/dense_6303/bias/v
}
*Adam/dense_6303/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6303/bias/v*
_output_shapes
:x*
dtype0
�
Adam/dense_6304/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*)
shared_nameAdam/dense_6304/kernel/v
�
,Adam/dense_6304/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6304/kernel/v*
_output_shapes

:x<*
dtype0
�
Adam/dense_6304/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_6304/bias/v
}
*Adam/dense_6304/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6304/bias/v*
_output_shapes
:<*
dtype0
�
Adam/dense_6305/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameAdam/dense_6305/kernel/v
�
,Adam/dense_6305/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6305/kernel/v*
_output_shapes

:<*
dtype0
�
Adam/dense_6305/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_6305/bias/v
}
*Adam/dense_6305/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6305/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�L
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�K
value�KB�K B�K
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
R
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
�
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem�m�m�m�&m�'m�4m�5m�:m�;m�Dm�Em�v�v�v�v�&v�'v�4v�5v�:v�;v�Dv�Ev�
 
V
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11
V
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11
�
Ometrics

Players
regularization_losses
Qlayer_regularization_losses
	variables
trainable_variables
Rnon_trainable_variables
Slayer_metrics
 
_]
VARIABLE_VALUEconv2d_10803/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_10803/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Tmetrics

Ulayers
regularization_losses
Vlayer_regularization_losses
	variables
trainable_variables
Wnon_trainable_variables
Xlayer_metrics
 
 
 
�
Ymetrics

Zlayers
regularization_losses
[layer_regularization_losses
	variables
trainable_variables
\non_trainable_variables
]layer_metrics
_]
VARIABLE_VALUEconv2d_10804/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_10804/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
^metrics

_layers
regularization_losses
`layer_regularization_losses
	variables
 trainable_variables
anon_trainable_variables
blayer_metrics
 
 
 
�
cmetrics

dlayers
"regularization_losses
elayer_regularization_losses
#	variables
$trainable_variables
fnon_trainable_variables
glayer_metrics
_]
VARIABLE_VALUEconv2d_10805/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_10805/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
�
hmetrics

ilayers
(regularization_losses
jlayer_regularization_losses
)	variables
*trainable_variables
knon_trainable_variables
llayer_metrics
 
 
 
�
mmetrics

nlayers
,regularization_losses
olayer_regularization_losses
-	variables
.trainable_variables
pnon_trainable_variables
qlayer_metrics
 
 
 
�
rmetrics

slayers
0regularization_losses
tlayer_regularization_losses
1	variables
2trainable_variables
unon_trainable_variables
vlayer_metrics
][
VARIABLE_VALUEdense_6303/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6303/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
�
wmetrics

xlayers
6regularization_losses
ylayer_regularization_losses
7	variables
8trainable_variables
znon_trainable_variables
{layer_metrics
][
VARIABLE_VALUEdense_6304/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6304/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
�
|metrics

}layers
<regularization_losses
~layer_regularization_losses
=	variables
>trainable_variables
non_trainable_variables
�layer_metrics
 
 
 
�
�metrics
�layers
@regularization_losses
 �layer_regularization_losses
A	variables
Btrainable_variables
�non_trainable_variables
�layer_metrics
][
VARIABLE_VALUEdense_6305/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6305/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
�
�metrics
�layers
Fregularization_losses
 �layer_regularization_losses
G	variables
Htrainable_variables
�non_trainable_variables
�layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUEAdam/conv2d_10803/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_10803/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_10804/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_10804/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_10805/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_10805/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_6303/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6303/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_6304/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6304/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_6305/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6305/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_10803/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_10803/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_10804/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_10804/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_10805/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_10805/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_6303/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6303/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_6304/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6304/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_6305/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6305/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
"serving_default_conv2d_10803_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conv2d_10803_inputconv2d_10803/kernelconv2d_10803/biasconv2d_10804/kernelconv2d_10804/biasconv2d_10805/kernelconv2d_10805/biasdense_6303/kerneldense_6303/biasdense_6304/kerneldense_6304/biasdense_6305/kerneldense_6305/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1614454
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'conv2d_10803/kernel/Read/ReadVariableOp%conv2d_10803/bias/Read/ReadVariableOp'conv2d_10804/kernel/Read/ReadVariableOp%conv2d_10804/bias/Read/ReadVariableOp'conv2d_10805/kernel/Read/ReadVariableOp%conv2d_10805/bias/Read/ReadVariableOp%dense_6303/kernel/Read/ReadVariableOp#dense_6303/bias/Read/ReadVariableOp%dense_6304/kernel/Read/ReadVariableOp#dense_6304/bias/Read/ReadVariableOp%dense_6305/kernel/Read/ReadVariableOp#dense_6305/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/conv2d_10803/kernel/m/Read/ReadVariableOp,Adam/conv2d_10803/bias/m/Read/ReadVariableOp.Adam/conv2d_10804/kernel/m/Read/ReadVariableOp,Adam/conv2d_10804/bias/m/Read/ReadVariableOp.Adam/conv2d_10805/kernel/m/Read/ReadVariableOp,Adam/conv2d_10805/bias/m/Read/ReadVariableOp,Adam/dense_6303/kernel/m/Read/ReadVariableOp*Adam/dense_6303/bias/m/Read/ReadVariableOp,Adam/dense_6304/kernel/m/Read/ReadVariableOp*Adam/dense_6304/bias/m/Read/ReadVariableOp,Adam/dense_6305/kernel/m/Read/ReadVariableOp*Adam/dense_6305/bias/m/Read/ReadVariableOp.Adam/conv2d_10803/kernel/v/Read/ReadVariableOp,Adam/conv2d_10803/bias/v/Read/ReadVariableOp.Adam/conv2d_10804/kernel/v/Read/ReadVariableOp,Adam/conv2d_10804/bias/v/Read/ReadVariableOp.Adam/conv2d_10805/kernel/v/Read/ReadVariableOp,Adam/conv2d_10805/bias/v/Read/ReadVariableOp,Adam/dense_6303/kernel/v/Read/ReadVariableOp*Adam/dense_6303/bias/v/Read/ReadVariableOp,Adam/dense_6304/kernel/v/Read/ReadVariableOp*Adam/dense_6304/bias/v/Read/ReadVariableOp,Adam/dense_6305/kernel/v/Read/ReadVariableOp*Adam/dense_6305/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_1614939
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10803/kernelconv2d_10803/biasconv2d_10804/kernelconv2d_10804/biasconv2d_10805/kernelconv2d_10805/biasdense_6303/kerneldense_6303/biasdense_6304/kerneldense_6304/biasdense_6305/kerneldense_6305/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_10803/kernel/mAdam/conv2d_10803/bias/mAdam/conv2d_10804/kernel/mAdam/conv2d_10804/bias/mAdam/conv2d_10805/kernel/mAdam/conv2d_10805/bias/mAdam/dense_6303/kernel/mAdam/dense_6303/bias/mAdam/dense_6304/kernel/mAdam/dense_6304/bias/mAdam/dense_6305/kernel/mAdam/dense_6305/bias/mAdam/conv2d_10803/kernel/vAdam/conv2d_10803/bias/vAdam/conv2d_10804/kernel/vAdam/conv2d_10804/bias/vAdam/conv2d_10805/kernel/vAdam/conv2d_10805/bias/vAdam/dense_6303/kernel/vAdam/dense_6303/bias/vAdam/dense_6304/kernel/vAdam/dense_6304/bias/vAdam/dense_6305/kernel/vAdam/dense_6305/bias/v*9
Tin2
02.*
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1615084�
�	
�
G__inference_dense_6304_layer_call_and_return_conditional_losses_1614165

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������<2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������x
 
_user_specified_nameinputs
�Y
�
"__inference__wrapped_model_1613989
conv2d_10803_input<
8sequential_1_conv2d_10803_conv2d_readvariableop_resource=
9sequential_1_conv2d_10803_biasadd_readvariableop_resource<
8sequential_1_conv2d_10804_conv2d_readvariableop_resource=
9sequential_1_conv2d_10804_biasadd_readvariableop_resource<
8sequential_1_conv2d_10805_conv2d_readvariableop_resource=
9sequential_1_conv2d_10805_biasadd_readvariableop_resource:
6sequential_1_dense_6303_matmul_readvariableop_resource;
7sequential_1_dense_6303_biasadd_readvariableop_resource:
6sequential_1_dense_6304_matmul_readvariableop_resource;
7sequential_1_dense_6304_biasadd_readvariableop_resource:
6sequential_1_dense_6305_matmul_readvariableop_resource;
7sequential_1_dense_6305_biasadd_readvariableop_resource
identity��0sequential_1/conv2d_10803/BiasAdd/ReadVariableOp�/sequential_1/conv2d_10803/Conv2D/ReadVariableOp�0sequential_1/conv2d_10804/BiasAdd/ReadVariableOp�/sequential_1/conv2d_10804/Conv2D/ReadVariableOp�0sequential_1/conv2d_10805/BiasAdd/ReadVariableOp�/sequential_1/conv2d_10805/Conv2D/ReadVariableOp�.sequential_1/dense_6303/BiasAdd/ReadVariableOp�-sequential_1/dense_6303/MatMul/ReadVariableOp�.sequential_1/dense_6304/BiasAdd/ReadVariableOp�-sequential_1/dense_6304/MatMul/ReadVariableOp�.sequential_1/dense_6305/BiasAdd/ReadVariableOp�-sequential_1/dense_6305/MatMul/ReadVariableOp�
/sequential_1/conv2d_10803/Conv2D/ReadVariableOpReadVariableOp8sequential_1_conv2d_10803_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/sequential_1/conv2d_10803/Conv2D/ReadVariableOp�
 sequential_1/conv2d_10803/Conv2DConv2Dconv2d_10803_input7sequential_1/conv2d_10803/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2"
 sequential_1/conv2d_10803/Conv2D�
0sequential_1/conv2d_10803/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_conv2d_10803_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential_1/conv2d_10803/BiasAdd/ReadVariableOp�
!sequential_1/conv2d_10803/BiasAddBiasAdd)sequential_1/conv2d_10803/Conv2D:output:08sequential_1/conv2d_10803/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2#
!sequential_1/conv2d_10803/BiasAdd�
sequential_1/conv2d_10803/ReluRelu*sequential_1/conv2d_10803/BiasAdd:output:0*
T0*1
_output_shapes
:����������� 2 
sequential_1/conv2d_10803/Relu�
'sequential_1/max_pooling2d_5403/MaxPoolMaxPool,sequential_1/conv2d_10803/Relu:activations:0*/
_output_shapes
:���������@@ *
ksize
*
paddingVALID*
strides
2)
'sequential_1/max_pooling2d_5403/MaxPool�
/sequential_1/conv2d_10804/Conv2D/ReadVariableOpReadVariableOp8sequential_1_conv2d_10804_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/sequential_1/conv2d_10804/Conv2D/ReadVariableOp�
 sequential_1/conv2d_10804/Conv2DConv2D0sequential_1/max_pooling2d_5403/MaxPool:output:07sequential_1/conv2d_10804/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
2"
 sequential_1/conv2d_10804/Conv2D�
0sequential_1/conv2d_10804/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_conv2d_10804_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential_1/conv2d_10804/BiasAdd/ReadVariableOp�
!sequential_1/conv2d_10804/BiasAddBiasAdd)sequential_1/conv2d_10804/Conv2D:output:08sequential_1/conv2d_10804/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@2#
!sequential_1/conv2d_10804/BiasAdd�
sequential_1/conv2d_10804/ReluRelu*sequential_1/conv2d_10804/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@2 
sequential_1/conv2d_10804/Relu�
'sequential_1/max_pooling2d_5404/MaxPoolMaxPool,sequential_1/conv2d_10804/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingVALID*
strides
2)
'sequential_1/max_pooling2d_5404/MaxPool�
/sequential_1/conv2d_10805/Conv2D/ReadVariableOpReadVariableOp8sequential_1_conv2d_10805_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype021
/sequential_1/conv2d_10805/Conv2D/ReadVariableOp�
 sequential_1/conv2d_10805/Conv2DConv2D0sequential_1/max_pooling2d_5404/MaxPool:output:07sequential_1/conv2d_10805/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
2"
 sequential_1/conv2d_10805/Conv2D�
0sequential_1/conv2d_10805/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_conv2d_10805_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0sequential_1/conv2d_10805/BiasAdd/ReadVariableOp�
!sequential_1/conv2d_10805/BiasAddBiasAdd)sequential_1/conv2d_10805/Conv2D:output:08sequential_1/conv2d_10805/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �2#
!sequential_1/conv2d_10805/BiasAdd�
sequential_1/conv2d_10805/ReluRelu*sequential_1/conv2d_10805/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �2 
sequential_1/conv2d_10805/Relu�
'sequential_1/max_pooling2d_5405/MaxPoolMaxPool,sequential_1/conv2d_10805/Relu:activations:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
2)
'sequential_1/max_pooling2d_5405/MaxPool�
sequential_1/flatten_1801/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  2!
sequential_1/flatten_1801/Const�
!sequential_1/flatten_1801/ReshapeReshape0sequential_1/max_pooling2d_5405/MaxPool:output:0(sequential_1/flatten_1801/Const:output:0*
T0*(
_output_shapes
:����������d2#
!sequential_1/flatten_1801/Reshape�
-sequential_1/dense_6303/MatMul/ReadVariableOpReadVariableOp6sequential_1_dense_6303_matmul_readvariableop_resource*
_output_shapes
:	�dx*
dtype02/
-sequential_1/dense_6303/MatMul/ReadVariableOp�
sequential_1/dense_6303/MatMulMatMul*sequential_1/flatten_1801/Reshape:output:05sequential_1/dense_6303/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2 
sequential_1/dense_6303/MatMul�
.sequential_1/dense_6303/BiasAdd/ReadVariableOpReadVariableOp7sequential_1_dense_6303_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype020
.sequential_1/dense_6303/BiasAdd/ReadVariableOp�
sequential_1/dense_6303/BiasAddBiasAdd(sequential_1/dense_6303/MatMul:product:06sequential_1/dense_6303/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2!
sequential_1/dense_6303/BiasAdd�
sequential_1/dense_6303/ReluRelu(sequential_1/dense_6303/BiasAdd:output:0*
T0*'
_output_shapes
:���������x2
sequential_1/dense_6303/Relu�
-sequential_1/dense_6304/MatMul/ReadVariableOpReadVariableOp6sequential_1_dense_6304_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02/
-sequential_1/dense_6304/MatMul/ReadVariableOp�
sequential_1/dense_6304/MatMulMatMul*sequential_1/dense_6303/Relu:activations:05sequential_1/dense_6304/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2 
sequential_1/dense_6304/MatMul�
.sequential_1/dense_6304/BiasAdd/ReadVariableOpReadVariableOp7sequential_1_dense_6304_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype020
.sequential_1/dense_6304/BiasAdd/ReadVariableOp�
sequential_1/dense_6304/BiasAddBiasAdd(sequential_1/dense_6304/MatMul:product:06sequential_1/dense_6304/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2!
sequential_1/dense_6304/BiasAdd�
sequential_1/dense_6304/ReluRelu(sequential_1/dense_6304/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
sequential_1/dense_6304/Relu�
sequential_1/dropout_1/IdentityIdentity*sequential_1/dense_6304/Relu:activations:0*
T0*'
_output_shapes
:���������<2!
sequential_1/dropout_1/Identity�
-sequential_1/dense_6305/MatMul/ReadVariableOpReadVariableOp6sequential_1_dense_6305_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02/
-sequential_1/dense_6305/MatMul/ReadVariableOp�
sequential_1/dense_6305/MatMulMatMul(sequential_1/dropout_1/Identity:output:05sequential_1/dense_6305/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_1/dense_6305/MatMul�
.sequential_1/dense_6305/BiasAdd/ReadVariableOpReadVariableOp7sequential_1_dense_6305_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_1/dense_6305/BiasAdd/ReadVariableOp�
sequential_1/dense_6305/BiasAddBiasAdd(sequential_1/dense_6305/MatMul:product:06sequential_1/dense_6305/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
sequential_1/dense_6305/BiasAdd�
sequential_1/dense_6305/SoftmaxSoftmax(sequential_1/dense_6305/BiasAdd:output:0*
T0*'
_output_shapes
:���������2!
sequential_1/dense_6305/Softmax�
IdentityIdentity)sequential_1/dense_6305/Softmax:softmax:01^sequential_1/conv2d_10803/BiasAdd/ReadVariableOp0^sequential_1/conv2d_10803/Conv2D/ReadVariableOp1^sequential_1/conv2d_10804/BiasAdd/ReadVariableOp0^sequential_1/conv2d_10804/Conv2D/ReadVariableOp1^sequential_1/conv2d_10805/BiasAdd/ReadVariableOp0^sequential_1/conv2d_10805/Conv2D/ReadVariableOp/^sequential_1/dense_6303/BiasAdd/ReadVariableOp.^sequential_1/dense_6303/MatMul/ReadVariableOp/^sequential_1/dense_6304/BiasAdd/ReadVariableOp.^sequential_1/dense_6304/MatMul/ReadVariableOp/^sequential_1/dense_6305/BiasAdd/ReadVariableOp.^sequential_1/dense_6305/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2d
0sequential_1/conv2d_10803/BiasAdd/ReadVariableOp0sequential_1/conv2d_10803/BiasAdd/ReadVariableOp2b
/sequential_1/conv2d_10803/Conv2D/ReadVariableOp/sequential_1/conv2d_10803/Conv2D/ReadVariableOp2d
0sequential_1/conv2d_10804/BiasAdd/ReadVariableOp0sequential_1/conv2d_10804/BiasAdd/ReadVariableOp2b
/sequential_1/conv2d_10804/Conv2D/ReadVariableOp/sequential_1/conv2d_10804/Conv2D/ReadVariableOp2d
0sequential_1/conv2d_10805/BiasAdd/ReadVariableOp0sequential_1/conv2d_10805/BiasAdd/ReadVariableOp2b
/sequential_1/conv2d_10805/Conv2D/ReadVariableOp/sequential_1/conv2d_10805/Conv2D/ReadVariableOp2`
.sequential_1/dense_6303/BiasAdd/ReadVariableOp.sequential_1/dense_6303/BiasAdd/ReadVariableOp2^
-sequential_1/dense_6303/MatMul/ReadVariableOp-sequential_1/dense_6303/MatMul/ReadVariableOp2`
.sequential_1/dense_6304/BiasAdd/ReadVariableOp.sequential_1/dense_6304/BiasAdd/ReadVariableOp2^
-sequential_1/dense_6304/MatMul/ReadVariableOp-sequential_1/dense_6304/MatMul/ReadVariableOp2`
.sequential_1/dense_6305/BiasAdd/ReadVariableOp.sequential_1/dense_6305/BiasAdd/ReadVariableOp2^
-sequential_1/dense_6305/MatMul/ReadVariableOp-sequential_1/dense_6305/MatMul/ReadVariableOp:e a
1
_output_shapes
:�����������
,
_user_specified_nameconv2d_10803_input
�
�
.__inference_conv2d_10804_layer_call_fn_1614663

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_16140682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
P
4__inference_max_pooling2d_5405_layer_call_fn_1614025

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
GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_16140192
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
.__inference_sequential_1_layer_call_fn_1614347
conv2d_10803_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_10803_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16143202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:�����������
,
_user_specified_nameconv2d_10803_input
�
e
I__inference_flatten_1801_layer_call_and_return_conditional_losses_1614689

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������d2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������

�:X T
0
_output_shapes
:���������

�
 
_user_specified_nameinputs
�	
�
.__inference_sequential_1_layer_call_fn_1614415
conv2d_10803_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_10803_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16143882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:�����������
,
_user_specified_nameconv2d_10803_input
�	
�
G__inference_dense_6305_layer_call_and_return_conditional_losses_1614772

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�

�
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_1614040

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
G__inference_dense_6305_layer_call_and_return_conditional_losses_1614222

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614751

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
d
+__inference_dropout_1_layer_call_fn_1614756

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16141932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������<22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�	
�
%__inference_signature_wrapper_1614454
conv2d_10803_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_10803_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_16139892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:�����������
,
_user_specified_nameconv2d_10803_input
�

�
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_1614068

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
k
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_1614019

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�P
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614513

inputs/
+conv2d_10803_conv2d_readvariableop_resource0
,conv2d_10803_biasadd_readvariableop_resource/
+conv2d_10804_conv2d_readvariableop_resource0
,conv2d_10804_biasadd_readvariableop_resource/
+conv2d_10805_conv2d_readvariableop_resource0
,conv2d_10805_biasadd_readvariableop_resource-
)dense_6303_matmul_readvariableop_resource.
*dense_6303_biasadd_readvariableop_resource-
)dense_6304_matmul_readvariableop_resource.
*dense_6304_biasadd_readvariableop_resource-
)dense_6305_matmul_readvariableop_resource.
*dense_6305_biasadd_readvariableop_resource
identity��#conv2d_10803/BiasAdd/ReadVariableOp�"conv2d_10803/Conv2D/ReadVariableOp�#conv2d_10804/BiasAdd/ReadVariableOp�"conv2d_10804/Conv2D/ReadVariableOp�#conv2d_10805/BiasAdd/ReadVariableOp�"conv2d_10805/Conv2D/ReadVariableOp�!dense_6303/BiasAdd/ReadVariableOp� dense_6303/MatMul/ReadVariableOp�!dense_6304/BiasAdd/ReadVariableOp� dense_6304/MatMul/ReadVariableOp�!dense_6305/BiasAdd/ReadVariableOp� dense_6305/MatMul/ReadVariableOp�
"conv2d_10803/Conv2D/ReadVariableOpReadVariableOp+conv2d_10803_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"conv2d_10803/Conv2D/ReadVariableOp�
conv2d_10803/Conv2DConv2Dinputs*conv2d_10803/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
conv2d_10803/Conv2D�
#conv2d_10803/BiasAdd/ReadVariableOpReadVariableOp,conv2d_10803_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv2d_10803/BiasAdd/ReadVariableOp�
conv2d_10803/BiasAddBiasAddconv2d_10803/Conv2D:output:0+conv2d_10803/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2
conv2d_10803/BiasAdd�
conv2d_10803/ReluReluconv2d_10803/BiasAdd:output:0*
T0*1
_output_shapes
:����������� 2
conv2d_10803/Relu�
max_pooling2d_5403/MaxPoolMaxPoolconv2d_10803/Relu:activations:0*/
_output_shapes
:���������@@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5403/MaxPool�
"conv2d_10804/Conv2D/ReadVariableOpReadVariableOp+conv2d_10804_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"conv2d_10804/Conv2D/ReadVariableOp�
conv2d_10804/Conv2DConv2D#max_pooling2d_5403/MaxPool:output:0*conv2d_10804/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
2
conv2d_10804/Conv2D�
#conv2d_10804/BiasAdd/ReadVariableOpReadVariableOp,conv2d_10804_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv2d_10804/BiasAdd/ReadVariableOp�
conv2d_10804/BiasAddBiasAddconv2d_10804/Conv2D:output:0+conv2d_10804/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@2
conv2d_10804/BiasAdd�
conv2d_10804/ReluReluconv2d_10804/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@2
conv2d_10804/Relu�
max_pooling2d_5404/MaxPoolMaxPoolconv2d_10804/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5404/MaxPool�
"conv2d_10805/Conv2D/ReadVariableOpReadVariableOp+conv2d_10805_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02$
"conv2d_10805/Conv2D/ReadVariableOp�
conv2d_10805/Conv2DConv2D#max_pooling2d_5404/MaxPool:output:0*conv2d_10805/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
2
conv2d_10805/Conv2D�
#conv2d_10805/BiasAdd/ReadVariableOpReadVariableOp,conv2d_10805_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#conv2d_10805/BiasAdd/ReadVariableOp�
conv2d_10805/BiasAddBiasAddconv2d_10805/Conv2D:output:0+conv2d_10805/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �2
conv2d_10805/BiasAdd�
conv2d_10805/ReluReluconv2d_10805/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �2
conv2d_10805/Relu�
max_pooling2d_5405/MaxPoolMaxPoolconv2d_10805/Relu:activations:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5405/MaxPooly
flatten_1801/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  2
flatten_1801/Const�
flatten_1801/ReshapeReshape#max_pooling2d_5405/MaxPool:output:0flatten_1801/Const:output:0*
T0*(
_output_shapes
:����������d2
flatten_1801/Reshape�
 dense_6303/MatMul/ReadVariableOpReadVariableOp)dense_6303_matmul_readvariableop_resource*
_output_shapes
:	�dx*
dtype02"
 dense_6303/MatMul/ReadVariableOp�
dense_6303/MatMulMatMulflatten_1801/Reshape:output:0(dense_6303/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2
dense_6303/MatMul�
!dense_6303/BiasAdd/ReadVariableOpReadVariableOp*dense_6303_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02#
!dense_6303/BiasAdd/ReadVariableOp�
dense_6303/BiasAddBiasAdddense_6303/MatMul:product:0)dense_6303/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2
dense_6303/BiasAddy
dense_6303/ReluReludense_6303/BiasAdd:output:0*
T0*'
_output_shapes
:���������x2
dense_6303/Relu�
 dense_6304/MatMul/ReadVariableOpReadVariableOp)dense_6304_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02"
 dense_6304/MatMul/ReadVariableOp�
dense_6304/MatMulMatMuldense_6303/Relu:activations:0(dense_6304/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_6304/MatMul�
!dense_6304/BiasAdd/ReadVariableOpReadVariableOp*dense_6304_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02#
!dense_6304/BiasAdd/ReadVariableOp�
dense_6304/BiasAddBiasAdddense_6304/MatMul:product:0)dense_6304/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_6304/BiasAddy
dense_6304/ReluReludense_6304/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
dense_6304/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const�
dropout_1/dropout/MulMuldense_6304/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������<2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShapedense_6304/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������<*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������<2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������<2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������<2
dropout_1/dropout/Mul_1�
 dense_6305/MatMul/ReadVariableOpReadVariableOp)dense_6305_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02"
 dense_6305/MatMul/ReadVariableOp�
dense_6305/MatMulMatMuldropout_1/dropout/Mul_1:z:0(dense_6305/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6305/MatMul�
!dense_6305/BiasAdd/ReadVariableOpReadVariableOp*dense_6305_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_6305/BiasAdd/ReadVariableOp�
dense_6305/BiasAddBiasAdddense_6305/MatMul:product:0)dense_6305/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6305/BiasAdd�
dense_6305/SoftmaxSoftmaxdense_6305/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_6305/Softmax�
IdentityIdentitydense_6305/Softmax:softmax:0$^conv2d_10803/BiasAdd/ReadVariableOp#^conv2d_10803/Conv2D/ReadVariableOp$^conv2d_10804/BiasAdd/ReadVariableOp#^conv2d_10804/Conv2D/ReadVariableOp$^conv2d_10805/BiasAdd/ReadVariableOp#^conv2d_10805/Conv2D/ReadVariableOp"^dense_6303/BiasAdd/ReadVariableOp!^dense_6303/MatMul/ReadVariableOp"^dense_6304/BiasAdd/ReadVariableOp!^dense_6304/MatMul/ReadVariableOp"^dense_6305/BiasAdd/ReadVariableOp!^dense_6305/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2J
#conv2d_10803/BiasAdd/ReadVariableOp#conv2d_10803/BiasAdd/ReadVariableOp2H
"conv2d_10803/Conv2D/ReadVariableOp"conv2d_10803/Conv2D/ReadVariableOp2J
#conv2d_10804/BiasAdd/ReadVariableOp#conv2d_10804/BiasAdd/ReadVariableOp2H
"conv2d_10804/Conv2D/ReadVariableOp"conv2d_10804/Conv2D/ReadVariableOp2J
#conv2d_10805/BiasAdd/ReadVariableOp#conv2d_10805/BiasAdd/ReadVariableOp2H
"conv2d_10805/Conv2D/ReadVariableOp"conv2d_10805/Conv2D/ReadVariableOp2F
!dense_6303/BiasAdd/ReadVariableOp!dense_6303/BiasAdd/ReadVariableOp2D
 dense_6303/MatMul/ReadVariableOp dense_6303/MatMul/ReadVariableOp2F
!dense_6304/BiasAdd/ReadVariableOp!dense_6304/BiasAdd/ReadVariableOp2D
 dense_6304/MatMul/ReadVariableOp dense_6304/MatMul/ReadVariableOp2F
!dense_6305/BiasAdd/ReadVariableOp!dense_6305/BiasAdd/ReadVariableOp2D
 dense_6305/MatMul/ReadVariableOp dense_6305/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�2
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614388

inputs
conv2d_10803_1614352
conv2d_10803_1614354
conv2d_10804_1614358
conv2d_10804_1614360
conv2d_10805_1614364
conv2d_10805_1614366
dense_6303_1614371
dense_6303_1614373
dense_6304_1614376
dense_6304_1614378
dense_6305_1614382
dense_6305_1614384
identity��$conv2d_10803/StatefulPartitionedCall�$conv2d_10804/StatefulPartitionedCall�$conv2d_10805/StatefulPartitionedCall�"dense_6303/StatefulPartitionedCall�"dense_6304/StatefulPartitionedCall�"dense_6305/StatefulPartitionedCall�
$conv2d_10803/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10803_1614352conv2d_10803_1614354*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_16140402&
$conv2d_10803/StatefulPartitionedCall�
"max_pooling2d_5403/PartitionedCallPartitionedCall-conv2d_10803/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_16139952$
"max_pooling2d_5403/PartitionedCall�
$conv2d_10804/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5403/PartitionedCall:output:0conv2d_10804_1614358conv2d_10804_1614360*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_16140682&
$conv2d_10804/StatefulPartitionedCall�
"max_pooling2d_5404/PartitionedCallPartitionedCall-conv2d_10804/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_16140072$
"max_pooling2d_5404/PartitionedCall�
$conv2d_10805/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5404/PartitionedCall:output:0conv2d_10805_1614364conv2d_10805_1614366*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_16140962&
$conv2d_10805/StatefulPartitionedCall�
"max_pooling2d_5405/PartitionedCallPartitionedCall-conv2d_10805/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_16140192$
"max_pooling2d_5405/PartitionedCall�
flatten_1801/PartitionedCallPartitionedCall+max_pooling2d_5405/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_flatten_1801_layer_call_and_return_conditional_losses_16141192
flatten_1801/PartitionedCall�
"dense_6303/StatefulPartitionedCallStatefulPartitionedCall%flatten_1801/PartitionedCall:output:0dense_6303_1614371dense_6303_1614373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6303_layer_call_and_return_conditional_losses_16141382$
"dense_6303/StatefulPartitionedCall�
"dense_6304/StatefulPartitionedCallStatefulPartitionedCall+dense_6303/StatefulPartitionedCall:output:0dense_6304_1614376dense_6304_1614378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6304_layer_call_and_return_conditional_losses_16141652$
"dense_6304/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall+dense_6304/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16141982
dropout_1/PartitionedCall�
"dense_6305/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_6305_1614382dense_6305_1614384*
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
GPU2*0J 8� *P
fKRI
G__inference_dense_6305_layer_call_and_return_conditional_losses_16142222$
"dense_6305/StatefulPartitionedCall�
IdentityIdentity+dense_6305/StatefulPartitionedCall:output:0%^conv2d_10803/StatefulPartitionedCall%^conv2d_10804/StatefulPartitionedCall%^conv2d_10805/StatefulPartitionedCall#^dense_6303/StatefulPartitionedCall#^dense_6304/StatefulPartitionedCall#^dense_6305/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2L
$conv2d_10803/StatefulPartitionedCall$conv2d_10803/StatefulPartitionedCall2L
$conv2d_10804/StatefulPartitionedCall$conv2d_10804/StatefulPartitionedCall2L
$conv2d_10805/StatefulPartitionedCall$conv2d_10805/StatefulPartitionedCall2H
"dense_6303/StatefulPartitionedCall"dense_6303/StatefulPartitionedCall2H
"dense_6304/StatefulPartitionedCall"dense_6304/StatefulPartitionedCall2H
"dense_6305/StatefulPartitionedCall"dense_6305/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�G
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614565

inputs/
+conv2d_10803_conv2d_readvariableop_resource0
,conv2d_10803_biasadd_readvariableop_resource/
+conv2d_10804_conv2d_readvariableop_resource0
,conv2d_10804_biasadd_readvariableop_resource/
+conv2d_10805_conv2d_readvariableop_resource0
,conv2d_10805_biasadd_readvariableop_resource-
)dense_6303_matmul_readvariableop_resource.
*dense_6303_biasadd_readvariableop_resource-
)dense_6304_matmul_readvariableop_resource.
*dense_6304_biasadd_readvariableop_resource-
)dense_6305_matmul_readvariableop_resource.
*dense_6305_biasadd_readvariableop_resource
identity��#conv2d_10803/BiasAdd/ReadVariableOp�"conv2d_10803/Conv2D/ReadVariableOp�#conv2d_10804/BiasAdd/ReadVariableOp�"conv2d_10804/Conv2D/ReadVariableOp�#conv2d_10805/BiasAdd/ReadVariableOp�"conv2d_10805/Conv2D/ReadVariableOp�!dense_6303/BiasAdd/ReadVariableOp� dense_6303/MatMul/ReadVariableOp�!dense_6304/BiasAdd/ReadVariableOp� dense_6304/MatMul/ReadVariableOp�!dense_6305/BiasAdd/ReadVariableOp� dense_6305/MatMul/ReadVariableOp�
"conv2d_10803/Conv2D/ReadVariableOpReadVariableOp+conv2d_10803_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"conv2d_10803/Conv2D/ReadVariableOp�
conv2d_10803/Conv2DConv2Dinputs*conv2d_10803/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
conv2d_10803/Conv2D�
#conv2d_10803/BiasAdd/ReadVariableOpReadVariableOp,conv2d_10803_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv2d_10803/BiasAdd/ReadVariableOp�
conv2d_10803/BiasAddBiasAddconv2d_10803/Conv2D:output:0+conv2d_10803/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2
conv2d_10803/BiasAdd�
conv2d_10803/ReluReluconv2d_10803/BiasAdd:output:0*
T0*1
_output_shapes
:����������� 2
conv2d_10803/Relu�
max_pooling2d_5403/MaxPoolMaxPoolconv2d_10803/Relu:activations:0*/
_output_shapes
:���������@@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5403/MaxPool�
"conv2d_10804/Conv2D/ReadVariableOpReadVariableOp+conv2d_10804_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"conv2d_10804/Conv2D/ReadVariableOp�
conv2d_10804/Conv2DConv2D#max_pooling2d_5403/MaxPool:output:0*conv2d_10804/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
2
conv2d_10804/Conv2D�
#conv2d_10804/BiasAdd/ReadVariableOpReadVariableOp,conv2d_10804_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv2d_10804/BiasAdd/ReadVariableOp�
conv2d_10804/BiasAddBiasAddconv2d_10804/Conv2D:output:0+conv2d_10804/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@2
conv2d_10804/BiasAdd�
conv2d_10804/ReluReluconv2d_10804/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@2
conv2d_10804/Relu�
max_pooling2d_5404/MaxPoolMaxPoolconv2d_10804/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5404/MaxPool�
"conv2d_10805/Conv2D/ReadVariableOpReadVariableOp+conv2d_10805_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02$
"conv2d_10805/Conv2D/ReadVariableOp�
conv2d_10805/Conv2DConv2D#max_pooling2d_5404/MaxPool:output:0*conv2d_10805/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
2
conv2d_10805/Conv2D�
#conv2d_10805/BiasAdd/ReadVariableOpReadVariableOp,conv2d_10805_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#conv2d_10805/BiasAdd/ReadVariableOp�
conv2d_10805/BiasAddBiasAddconv2d_10805/Conv2D:output:0+conv2d_10805/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �2
conv2d_10805/BiasAdd�
conv2d_10805/ReluReluconv2d_10805/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �2
conv2d_10805/Relu�
max_pooling2d_5405/MaxPoolMaxPoolconv2d_10805/Relu:activations:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5405/MaxPooly
flatten_1801/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  2
flatten_1801/Const�
flatten_1801/ReshapeReshape#max_pooling2d_5405/MaxPool:output:0flatten_1801/Const:output:0*
T0*(
_output_shapes
:����������d2
flatten_1801/Reshape�
 dense_6303/MatMul/ReadVariableOpReadVariableOp)dense_6303_matmul_readvariableop_resource*
_output_shapes
:	�dx*
dtype02"
 dense_6303/MatMul/ReadVariableOp�
dense_6303/MatMulMatMulflatten_1801/Reshape:output:0(dense_6303/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2
dense_6303/MatMul�
!dense_6303/BiasAdd/ReadVariableOpReadVariableOp*dense_6303_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02#
!dense_6303/BiasAdd/ReadVariableOp�
dense_6303/BiasAddBiasAdddense_6303/MatMul:product:0)dense_6303/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2
dense_6303/BiasAddy
dense_6303/ReluReludense_6303/BiasAdd:output:0*
T0*'
_output_shapes
:���������x2
dense_6303/Relu�
 dense_6304/MatMul/ReadVariableOpReadVariableOp)dense_6304_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02"
 dense_6304/MatMul/ReadVariableOp�
dense_6304/MatMulMatMuldense_6303/Relu:activations:0(dense_6304/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_6304/MatMul�
!dense_6304/BiasAdd/ReadVariableOpReadVariableOp*dense_6304_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02#
!dense_6304/BiasAdd/ReadVariableOp�
dense_6304/BiasAddBiasAdddense_6304/MatMul:product:0)dense_6304/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_6304/BiasAddy
dense_6304/ReluReludense_6304/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
dense_6304/Relu�
dropout_1/IdentityIdentitydense_6304/Relu:activations:0*
T0*'
_output_shapes
:���������<2
dropout_1/Identity�
 dense_6305/MatMul/ReadVariableOpReadVariableOp)dense_6305_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02"
 dense_6305/MatMul/ReadVariableOp�
dense_6305/MatMulMatMuldropout_1/Identity:output:0(dense_6305/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6305/MatMul�
!dense_6305/BiasAdd/ReadVariableOpReadVariableOp*dense_6305_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_6305/BiasAdd/ReadVariableOp�
dense_6305/BiasAddBiasAdddense_6305/MatMul:product:0)dense_6305/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6305/BiasAdd�
dense_6305/SoftmaxSoftmaxdense_6305/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_6305/Softmax�
IdentityIdentitydense_6305/Softmax:softmax:0$^conv2d_10803/BiasAdd/ReadVariableOp#^conv2d_10803/Conv2D/ReadVariableOp$^conv2d_10804/BiasAdd/ReadVariableOp#^conv2d_10804/Conv2D/ReadVariableOp$^conv2d_10805/BiasAdd/ReadVariableOp#^conv2d_10805/Conv2D/ReadVariableOp"^dense_6303/BiasAdd/ReadVariableOp!^dense_6303/MatMul/ReadVariableOp"^dense_6304/BiasAdd/ReadVariableOp!^dense_6304/MatMul/ReadVariableOp"^dense_6305/BiasAdd/ReadVariableOp!^dense_6305/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2J
#conv2d_10803/BiasAdd/ReadVariableOp#conv2d_10803/BiasAdd/ReadVariableOp2H
"conv2d_10803/Conv2D/ReadVariableOp"conv2d_10803/Conv2D/ReadVariableOp2J
#conv2d_10804/BiasAdd/ReadVariableOp#conv2d_10804/BiasAdd/ReadVariableOp2H
"conv2d_10804/Conv2D/ReadVariableOp"conv2d_10804/Conv2D/ReadVariableOp2J
#conv2d_10805/BiasAdd/ReadVariableOp#conv2d_10805/BiasAdd/ReadVariableOp2H
"conv2d_10805/Conv2D/ReadVariableOp"conv2d_10805/Conv2D/ReadVariableOp2F
!dense_6303/BiasAdd/ReadVariableOp!dense_6303/BiasAdd/ReadVariableOp2D
 dense_6303/MatMul/ReadVariableOp dense_6303/MatMul/ReadVariableOp2F
!dense_6304/BiasAdd/ReadVariableOp!dense_6304/BiasAdd/ReadVariableOp2D
 dense_6304/MatMul/ReadVariableOp dense_6304/MatMul/ReadVariableOp2F
!dense_6305/BiasAdd/ReadVariableOp!dense_6305/BiasAdd/ReadVariableOp2D
 dense_6305/MatMul/ReadVariableOp dense_6305/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
,__inference_dense_6305_layer_call_fn_1614781

inputs
unknown
	unknown_0
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
GPU2*0J 8� *P
fKRI
G__inference_dense_6305_layer_call_and_return_conditional_losses_16142222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�	
�
.__inference_sequential_1_layer_call_fn_1614623

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16143882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_flatten_1801_layer_call_and_return_conditional_losses_1614119

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������d2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������

�:X T
0
_output_shapes
:���������

�
 
_user_specified_nameinputs
�4
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614320

inputs
conv2d_10803_1614284
conv2d_10803_1614286
conv2d_10804_1614290
conv2d_10804_1614292
conv2d_10805_1614296
conv2d_10805_1614298
dense_6303_1614303
dense_6303_1614305
dense_6304_1614308
dense_6304_1614310
dense_6305_1614314
dense_6305_1614316
identity��$conv2d_10803/StatefulPartitionedCall�$conv2d_10804/StatefulPartitionedCall�$conv2d_10805/StatefulPartitionedCall�"dense_6303/StatefulPartitionedCall�"dense_6304/StatefulPartitionedCall�"dense_6305/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
$conv2d_10803/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10803_1614284conv2d_10803_1614286*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_16140402&
$conv2d_10803/StatefulPartitionedCall�
"max_pooling2d_5403/PartitionedCallPartitionedCall-conv2d_10803/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_16139952$
"max_pooling2d_5403/PartitionedCall�
$conv2d_10804/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5403/PartitionedCall:output:0conv2d_10804_1614290conv2d_10804_1614292*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_16140682&
$conv2d_10804/StatefulPartitionedCall�
"max_pooling2d_5404/PartitionedCallPartitionedCall-conv2d_10804/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_16140072$
"max_pooling2d_5404/PartitionedCall�
$conv2d_10805/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5404/PartitionedCall:output:0conv2d_10805_1614296conv2d_10805_1614298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_16140962&
$conv2d_10805/StatefulPartitionedCall�
"max_pooling2d_5405/PartitionedCallPartitionedCall-conv2d_10805/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_16140192$
"max_pooling2d_5405/PartitionedCall�
flatten_1801/PartitionedCallPartitionedCall+max_pooling2d_5405/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_flatten_1801_layer_call_and_return_conditional_losses_16141192
flatten_1801/PartitionedCall�
"dense_6303/StatefulPartitionedCallStatefulPartitionedCall%flatten_1801/PartitionedCall:output:0dense_6303_1614303dense_6303_1614305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6303_layer_call_and_return_conditional_losses_16141382$
"dense_6303/StatefulPartitionedCall�
"dense_6304/StatefulPartitionedCallStatefulPartitionedCall+dense_6303/StatefulPartitionedCall:output:0dense_6304_1614308dense_6304_1614310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6304_layer_call_and_return_conditional_losses_16141652$
"dense_6304/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall+dense_6304/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16141932#
!dropout_1/StatefulPartitionedCall�
"dense_6305/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_6305_1614314dense_6305_1614316*
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
GPU2*0J 8� *P
fKRI
G__inference_dense_6305_layer_call_and_return_conditional_losses_16142222$
"dense_6305/StatefulPartitionedCall�
IdentityIdentity+dense_6305/StatefulPartitionedCall:output:0%^conv2d_10803/StatefulPartitionedCall%^conv2d_10804/StatefulPartitionedCall%^conv2d_10805/StatefulPartitionedCall#^dense_6303/StatefulPartitionedCall#^dense_6304/StatefulPartitionedCall#^dense_6305/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2L
$conv2d_10803/StatefulPartitionedCall$conv2d_10803/StatefulPartitionedCall2L
$conv2d_10804/StatefulPartitionedCall$conv2d_10804/StatefulPartitionedCall2L
$conv2d_10805/StatefulPartitionedCall$conv2d_10805/StatefulPartitionedCall2H
"dense_6303/StatefulPartitionedCall"dense_6303/StatefulPartitionedCall2H
"dense_6304/StatefulPartitionedCall"dense_6304/StatefulPartitionedCall2H
"dense_6305/StatefulPartitionedCall"dense_6305/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�_
�
 __inference__traced_save_1614939
file_prefix2
.savev2_conv2d_10803_kernel_read_readvariableop0
,savev2_conv2d_10803_bias_read_readvariableop2
.savev2_conv2d_10804_kernel_read_readvariableop0
,savev2_conv2d_10804_bias_read_readvariableop2
.savev2_conv2d_10805_kernel_read_readvariableop0
,savev2_conv2d_10805_bias_read_readvariableop0
,savev2_dense_6303_kernel_read_readvariableop.
*savev2_dense_6303_bias_read_readvariableop0
,savev2_dense_6304_kernel_read_readvariableop.
*savev2_dense_6304_bias_read_readvariableop0
,savev2_dense_6305_kernel_read_readvariableop.
*savev2_dense_6305_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_conv2d_10803_kernel_m_read_readvariableop7
3savev2_adam_conv2d_10803_bias_m_read_readvariableop9
5savev2_adam_conv2d_10804_kernel_m_read_readvariableop7
3savev2_adam_conv2d_10804_bias_m_read_readvariableop9
5savev2_adam_conv2d_10805_kernel_m_read_readvariableop7
3savev2_adam_conv2d_10805_bias_m_read_readvariableop7
3savev2_adam_dense_6303_kernel_m_read_readvariableop5
1savev2_adam_dense_6303_bias_m_read_readvariableop7
3savev2_adam_dense_6304_kernel_m_read_readvariableop5
1savev2_adam_dense_6304_bias_m_read_readvariableop7
3savev2_adam_dense_6305_kernel_m_read_readvariableop5
1savev2_adam_dense_6305_bias_m_read_readvariableop9
5savev2_adam_conv2d_10803_kernel_v_read_readvariableop7
3savev2_adam_conv2d_10803_bias_v_read_readvariableop9
5savev2_adam_conv2d_10804_kernel_v_read_readvariableop7
3savev2_adam_conv2d_10804_bias_v_read_readvariableop9
5savev2_adam_conv2d_10805_kernel_v_read_readvariableop7
3savev2_adam_conv2d_10805_bias_v_read_readvariableop7
3savev2_adam_dense_6303_kernel_v_read_readvariableop5
1savev2_adam_dense_6303_bias_v_read_readvariableop7
3savev2_adam_dense_6304_kernel_v_read_readvariableop5
1savev2_adam_dense_6304_bias_v_read_readvariableop7
3savev2_adam_dense_6305_kernel_v_read_readvariableop5
1savev2_adam_dense_6305_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*�
value�B�.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv2d_10803_kernel_read_readvariableop,savev2_conv2d_10803_bias_read_readvariableop.savev2_conv2d_10804_kernel_read_readvariableop,savev2_conv2d_10804_bias_read_readvariableop.savev2_conv2d_10805_kernel_read_readvariableop,savev2_conv2d_10805_bias_read_readvariableop,savev2_dense_6303_kernel_read_readvariableop*savev2_dense_6303_bias_read_readvariableop,savev2_dense_6304_kernel_read_readvariableop*savev2_dense_6304_bias_read_readvariableop,savev2_dense_6305_kernel_read_readvariableop*savev2_dense_6305_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_conv2d_10803_kernel_m_read_readvariableop3savev2_adam_conv2d_10803_bias_m_read_readvariableop5savev2_adam_conv2d_10804_kernel_m_read_readvariableop3savev2_adam_conv2d_10804_bias_m_read_readvariableop5savev2_adam_conv2d_10805_kernel_m_read_readvariableop3savev2_adam_conv2d_10805_bias_m_read_readvariableop3savev2_adam_dense_6303_kernel_m_read_readvariableop1savev2_adam_dense_6303_bias_m_read_readvariableop3savev2_adam_dense_6304_kernel_m_read_readvariableop1savev2_adam_dense_6304_bias_m_read_readvariableop3savev2_adam_dense_6305_kernel_m_read_readvariableop1savev2_adam_dense_6305_bias_m_read_readvariableop5savev2_adam_conv2d_10803_kernel_v_read_readvariableop3savev2_adam_conv2d_10803_bias_v_read_readvariableop5savev2_adam_conv2d_10804_kernel_v_read_readvariableop3savev2_adam_conv2d_10804_bias_v_read_readvariableop5savev2_adam_conv2d_10805_kernel_v_read_readvariableop3savev2_adam_conv2d_10805_bias_v_read_readvariableop3savev2_adam_dense_6303_kernel_v_read_readvariableop1savev2_adam_dense_6303_bias_v_read_readvariableop3savev2_adam_dense_6304_kernel_v_read_readvariableop1savev2_adam_dense_6304_bias_v_read_readvariableop3savev2_adam_dense_6305_kernel_v_read_readvariableop1savev2_adam_dense_6305_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:@�:�:	�dx:x:x<:<:<:: : : : : : : : : : : : @:@:@�:�:	�dx:x:x<:<:<:: : : @:@:@�:�:	�dx:x:x<:<:<:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:%!

_output_shapes
:	�dx: 

_output_shapes
:x:$	 

_output_shapes

:x<: 


_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:%!

_output_shapes
:	�dx: 

_output_shapes
:x:$ 

_output_shapes

:x<: 

_output_shapes
:<:$  

_output_shapes

:<: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@�:!'

_output_shapes	
:�:%(!

_output_shapes
:	�dx: )

_output_shapes
:x:$* 

_output_shapes

:x<: +

_output_shapes
:<:$, 

_output_shapes

:<: -

_output_shapes
::.

_output_shapes
: 
��
�
#__inference__traced_restore_1615084
file_prefix(
$assignvariableop_conv2d_10803_kernel(
$assignvariableop_1_conv2d_10803_bias*
&assignvariableop_2_conv2d_10804_kernel(
$assignvariableop_3_conv2d_10804_bias*
&assignvariableop_4_conv2d_10805_kernel(
$assignvariableop_5_conv2d_10805_bias(
$assignvariableop_6_dense_6303_kernel&
"assignvariableop_7_dense_6303_bias(
$assignvariableop_8_dense_6304_kernel&
"assignvariableop_9_dense_6304_bias)
%assignvariableop_10_dense_6305_kernel'
#assignvariableop_11_dense_6305_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_12
.assignvariableop_21_adam_conv2d_10803_kernel_m0
,assignvariableop_22_adam_conv2d_10803_bias_m2
.assignvariableop_23_adam_conv2d_10804_kernel_m0
,assignvariableop_24_adam_conv2d_10804_bias_m2
.assignvariableop_25_adam_conv2d_10805_kernel_m0
,assignvariableop_26_adam_conv2d_10805_bias_m0
,assignvariableop_27_adam_dense_6303_kernel_m.
*assignvariableop_28_adam_dense_6303_bias_m0
,assignvariableop_29_adam_dense_6304_kernel_m.
*assignvariableop_30_adam_dense_6304_bias_m0
,assignvariableop_31_adam_dense_6305_kernel_m.
*assignvariableop_32_adam_dense_6305_bias_m2
.assignvariableop_33_adam_conv2d_10803_kernel_v0
,assignvariableop_34_adam_conv2d_10803_bias_v2
.assignvariableop_35_adam_conv2d_10804_kernel_v0
,assignvariableop_36_adam_conv2d_10804_bias_v2
.assignvariableop_37_adam_conv2d_10805_kernel_v0
,assignvariableop_38_adam_conv2d_10805_bias_v0
,assignvariableop_39_adam_dense_6303_kernel_v.
*assignvariableop_40_adam_dense_6303_bias_v0
,assignvariableop_41_adam_dense_6304_kernel_v.
*assignvariableop_42_adam_dense_6304_bias_v0
,assignvariableop_43_adam_dense_6305_kernel_v.
*assignvariableop_44_adam_dense_6305_bias_v
identity_46��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*�
value�B�.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp$assignvariableop_conv2d_10803_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_10803_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv2d_10804_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv2d_10804_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv2d_10805_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv2d_10805_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_6303_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_6303_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_dense_6304_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_6304_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp%assignvariableop_10_dense_6305_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_6305_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_conv2d_10803_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_conv2d_10803_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_conv2d_10804_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_conv2d_10804_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_conv2d_10805_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_conv2d_10805_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_dense_6303_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_6303_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_dense_6304_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_6304_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_dense_6305_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_6305_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_conv2d_10803_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv2d_10803_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_conv2d_10804_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv2d_10804_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_conv2d_10805_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_conv2d_10805_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_dense_6303_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_6303_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_dense_6304_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_6304_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_dense_6305_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_6305_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45�
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614746

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�

�
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_1614674

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������  �2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�	
�
G__inference_dense_6303_layer_call_and_return_conditional_losses_1614138

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�dx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������x2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������x2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������d
 
_user_specified_nameinputs
�
k
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_1614007

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
k
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_1613995

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614198

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
J
.__inference_flatten_1801_layer_call_fn_1614694

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
:����������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_flatten_1801_layer_call_and_return_conditional_losses_16141192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������

�:X T
0
_output_shapes
:���������

�
 
_user_specified_nameinputs
�
�
.__inference_conv2d_10803_layer_call_fn_1614643

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_16140402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
G__inference_dense_6303_layer_call_and_return_conditional_losses_1614705

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�dx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������x2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������x2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������d
 
_user_specified_nameinputs
�
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614193

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�	
�
.__inference_sequential_1_layer_call_fn_1614594

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16143202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�2
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614278
conv2d_10803_input
conv2d_10803_1614242
conv2d_10803_1614244
conv2d_10804_1614248
conv2d_10804_1614250
conv2d_10805_1614254
conv2d_10805_1614256
dense_6303_1614261
dense_6303_1614263
dense_6304_1614266
dense_6304_1614268
dense_6305_1614272
dense_6305_1614274
identity��$conv2d_10803/StatefulPartitionedCall�$conv2d_10804/StatefulPartitionedCall�$conv2d_10805/StatefulPartitionedCall�"dense_6303/StatefulPartitionedCall�"dense_6304/StatefulPartitionedCall�"dense_6305/StatefulPartitionedCall�
$conv2d_10803/StatefulPartitionedCallStatefulPartitionedCallconv2d_10803_inputconv2d_10803_1614242conv2d_10803_1614244*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_16140402&
$conv2d_10803/StatefulPartitionedCall�
"max_pooling2d_5403/PartitionedCallPartitionedCall-conv2d_10803/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_16139952$
"max_pooling2d_5403/PartitionedCall�
$conv2d_10804/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5403/PartitionedCall:output:0conv2d_10804_1614248conv2d_10804_1614250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_16140682&
$conv2d_10804/StatefulPartitionedCall�
"max_pooling2d_5404/PartitionedCallPartitionedCall-conv2d_10804/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_16140072$
"max_pooling2d_5404/PartitionedCall�
$conv2d_10805/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5404/PartitionedCall:output:0conv2d_10805_1614254conv2d_10805_1614256*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_16140962&
$conv2d_10805/StatefulPartitionedCall�
"max_pooling2d_5405/PartitionedCallPartitionedCall-conv2d_10805/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_16140192$
"max_pooling2d_5405/PartitionedCall�
flatten_1801/PartitionedCallPartitionedCall+max_pooling2d_5405/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_flatten_1801_layer_call_and_return_conditional_losses_16141192
flatten_1801/PartitionedCall�
"dense_6303/StatefulPartitionedCallStatefulPartitionedCall%flatten_1801/PartitionedCall:output:0dense_6303_1614261dense_6303_1614263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6303_layer_call_and_return_conditional_losses_16141382$
"dense_6303/StatefulPartitionedCall�
"dense_6304/StatefulPartitionedCallStatefulPartitionedCall+dense_6303/StatefulPartitionedCall:output:0dense_6304_1614266dense_6304_1614268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6304_layer_call_and_return_conditional_losses_16141652$
"dense_6304/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall+dense_6304/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16141982
dropout_1/PartitionedCall�
"dense_6305/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_6305_1614272dense_6305_1614274*
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
GPU2*0J 8� *P
fKRI
G__inference_dense_6305_layer_call_and_return_conditional_losses_16142222$
"dense_6305/StatefulPartitionedCall�
IdentityIdentity+dense_6305/StatefulPartitionedCall:output:0%^conv2d_10803/StatefulPartitionedCall%^conv2d_10804/StatefulPartitionedCall%^conv2d_10805/StatefulPartitionedCall#^dense_6303/StatefulPartitionedCall#^dense_6304/StatefulPartitionedCall#^dense_6305/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2L
$conv2d_10803/StatefulPartitionedCall$conv2d_10803/StatefulPartitionedCall2L
$conv2d_10804/StatefulPartitionedCall$conv2d_10804/StatefulPartitionedCall2L
$conv2d_10805/StatefulPartitionedCall$conv2d_10805/StatefulPartitionedCall2H
"dense_6303/StatefulPartitionedCall"dense_6303/StatefulPartitionedCall2H
"dense_6304/StatefulPartitionedCall"dense_6304/StatefulPartitionedCall2H
"dense_6305/StatefulPartitionedCall"dense_6305/StatefulPartitionedCall:e a
1
_output_shapes
:�����������
,
_user_specified_nameconv2d_10803_input
�

�
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_1614634

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
P
4__inference_max_pooling2d_5403_layer_call_fn_1614001

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
GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_16139952
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
.__inference_conv2d_10805_layer_call_fn_1614683

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_16140962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������  �2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
G
+__inference_dropout_1_layer_call_fn_1614761

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
:���������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16141982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�4
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614239
conv2d_10803_input
conv2d_10803_1614051
conv2d_10803_1614053
conv2d_10804_1614079
conv2d_10804_1614081
conv2d_10805_1614107
conv2d_10805_1614109
dense_6303_1614149
dense_6303_1614151
dense_6304_1614176
dense_6304_1614178
dense_6305_1614233
dense_6305_1614235
identity��$conv2d_10803/StatefulPartitionedCall�$conv2d_10804/StatefulPartitionedCall�$conv2d_10805/StatefulPartitionedCall�"dense_6303/StatefulPartitionedCall�"dense_6304/StatefulPartitionedCall�"dense_6305/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
$conv2d_10803/StatefulPartitionedCallStatefulPartitionedCallconv2d_10803_inputconv2d_10803_1614051conv2d_10803_1614053*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_16140402&
$conv2d_10803/StatefulPartitionedCall�
"max_pooling2d_5403/PartitionedCallPartitionedCall-conv2d_10803/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_16139952$
"max_pooling2d_5403/PartitionedCall�
$conv2d_10804/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5403/PartitionedCall:output:0conv2d_10804_1614079conv2d_10804_1614081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_16140682&
$conv2d_10804/StatefulPartitionedCall�
"max_pooling2d_5404/PartitionedCallPartitionedCall-conv2d_10804/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_16140072$
"max_pooling2d_5404/PartitionedCall�
$conv2d_10805/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_5404/PartitionedCall:output:0conv2d_10805_1614107conv2d_10805_1614109*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_16140962&
$conv2d_10805/StatefulPartitionedCall�
"max_pooling2d_5405/PartitionedCallPartitionedCall-conv2d_10805/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_16140192$
"max_pooling2d_5405/PartitionedCall�
flatten_1801/PartitionedCallPartitionedCall+max_pooling2d_5405/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_flatten_1801_layer_call_and_return_conditional_losses_16141192
flatten_1801/PartitionedCall�
"dense_6303/StatefulPartitionedCallStatefulPartitionedCall%flatten_1801/PartitionedCall:output:0dense_6303_1614149dense_6303_1614151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6303_layer_call_and_return_conditional_losses_16141382$
"dense_6303/StatefulPartitionedCall�
"dense_6304/StatefulPartitionedCallStatefulPartitionedCall+dense_6303/StatefulPartitionedCall:output:0dense_6304_1614176dense_6304_1614178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6304_layer_call_and_return_conditional_losses_16141652$
"dense_6304/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall+dense_6304/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16141932#
!dropout_1/StatefulPartitionedCall�
"dense_6305/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_6305_1614233dense_6305_1614235*
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
GPU2*0J 8� *P
fKRI
G__inference_dense_6305_layer_call_and_return_conditional_losses_16142222$
"dense_6305/StatefulPartitionedCall�
IdentityIdentity+dense_6305/StatefulPartitionedCall:output:0%^conv2d_10803/StatefulPartitionedCall%^conv2d_10804/StatefulPartitionedCall%^conv2d_10805/StatefulPartitionedCall#^dense_6303/StatefulPartitionedCall#^dense_6304/StatefulPartitionedCall#^dense_6305/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2L
$conv2d_10803/StatefulPartitionedCall$conv2d_10803/StatefulPartitionedCall2L
$conv2d_10804/StatefulPartitionedCall$conv2d_10804/StatefulPartitionedCall2L
$conv2d_10805/StatefulPartitionedCall$conv2d_10805/StatefulPartitionedCall2H
"dense_6303/StatefulPartitionedCall"dense_6303/StatefulPartitionedCall2H
"dense_6304/StatefulPartitionedCall"dense_6304/StatefulPartitionedCall2H
"dense_6305/StatefulPartitionedCall"dense_6305/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:e a
1
_output_shapes
:�����������
,
_user_specified_nameconv2d_10803_input
�

�
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_1614096

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������  �2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
P
4__inference_max_pooling2d_5404_layer_call_fn_1614013

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
GPU2*0J 8� *X
fSRQ
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_16140072
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_dense_6303_layer_call_fn_1614714

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6303_layer_call_and_return_conditional_losses_16141382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������x2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������d::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
,__inference_dense_6304_layer_call_fn_1614734

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_6304_layer_call_and_return_conditional_losses_16141652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������x
 
_user_specified_nameinputs
�	
�
G__inference_dense_6304_layer_call_and_return_conditional_losses_1614725

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������<2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������x
 
_user_specified_nameinputs
�

�
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_1614654

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
[
conv2d_10803_inputE
$serving_default_conv2d_10803_input:0�����������>

dense_63050
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�V
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�R
_tf_keras_sequential�R{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10803_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10803", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5403", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10804", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5404", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10805", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5405", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_1801", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6303", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_6304", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6305", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10803_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10803", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5403", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10804", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5404", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10805", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5405", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_1801", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6303", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_6304", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6305", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_10803", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10803", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
�
regularization_losses
	variables
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_5403", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5403", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_10804", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10804", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
�
"regularization_losses
#	variables
$trainable_variables
%	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_5404", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5404", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_10805", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10805", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
�
,regularization_losses
-	variables
.trainable_variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_5405", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5405", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
0regularization_losses
1	variables
2trainable_variables
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1801", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1801", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6303", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6303", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12800]}}
�

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6304", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6304", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
�
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6305", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6305", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
�
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem�m�m�m�&m�'m�4m�5m�:m�;m�Dm�Em�v�v�v�v�&v�'v�4v�5v�:v�;v�Dv�Ev�"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11"
trackable_list_wrapper
v
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11"
trackable_list_wrapper
�
Ometrics

Players
regularization_losses
Qlayer_regularization_losses
	variables
trainable_variables
Rnon_trainable_variables
Slayer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
-:+ 2conv2d_10803/kernel
: 2conv2d_10803/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Tmetrics

Ulayers
regularization_losses
Vlayer_regularization_losses
	variables
trainable_variables
Wnon_trainable_variables
Xlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ymetrics

Zlayers
regularization_losses
[layer_regularization_losses
	variables
trainable_variables
\non_trainable_variables
]layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+ @2conv2d_10804/kernel
:@2conv2d_10804/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
^metrics

_layers
regularization_losses
`layer_regularization_losses
	variables
 trainable_variables
anon_trainable_variables
blayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cmetrics

dlayers
"regularization_losses
elayer_regularization_losses
#	variables
$trainable_variables
fnon_trainable_variables
glayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,@�2conv2d_10805/kernel
 :�2conv2d_10805/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
hmetrics

ilayers
(regularization_losses
jlayer_regularization_losses
)	variables
*trainable_variables
knon_trainable_variables
llayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mmetrics

nlayers
,regularization_losses
olayer_regularization_losses
-	variables
.trainable_variables
pnon_trainable_variables
qlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rmetrics

slayers
0regularization_losses
tlayer_regularization_losses
1	variables
2trainable_variables
unon_trainable_variables
vlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"	�dx2dense_6303/kernel
:x2dense_6303/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
wmetrics

xlayers
6regularization_losses
ylayer_regularization_losses
7	variables
8trainable_variables
znon_trainable_variables
{layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!x<2dense_6304/kernel
:<2dense_6304/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
�
|metrics

}layers
<regularization_losses
~layer_regularization_losses
=	variables
>trainable_variables
non_trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layers
@regularization_losses
 �layer_regularization_losses
A	variables
Btrainable_variables
�non_trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!<2dense_6305/kernel
:2dense_6305/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
�
�metrics
�layers
Fregularization_losses
 �layer_regularization_losses
G	variables
Htrainable_variables
�non_trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
�0
�1"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
2:0 2Adam/conv2d_10803/kernel/m
$:" 2Adam/conv2d_10803/bias/m
2:0 @2Adam/conv2d_10804/kernel/m
$:"@2Adam/conv2d_10804/bias/m
3:1@�2Adam/conv2d_10805/kernel/m
%:#�2Adam/conv2d_10805/bias/m
):'	�dx2Adam/dense_6303/kernel/m
": x2Adam/dense_6303/bias/m
(:&x<2Adam/dense_6304/kernel/m
": <2Adam/dense_6304/bias/m
(:&<2Adam/dense_6305/kernel/m
": 2Adam/dense_6305/bias/m
2:0 2Adam/conv2d_10803/kernel/v
$:" 2Adam/conv2d_10803/bias/v
2:0 @2Adam/conv2d_10804/kernel/v
$:"@2Adam/conv2d_10804/bias/v
3:1@�2Adam/conv2d_10805/kernel/v
%:#�2Adam/conv2d_10805/bias/v
):'	�dx2Adam/dense_6303/kernel/v
": x2Adam/dense_6303/bias/v
(:&x<2Adam/dense_6304/kernel/v
": <2Adam/dense_6304/bias/v
(:&<2Adam/dense_6305/kernel/v
": 2Adam/dense_6305/bias/v
�2�
.__inference_sequential_1_layer_call_fn_1614623
.__inference_sequential_1_layer_call_fn_1614415
.__inference_sequential_1_layer_call_fn_1614594
.__inference_sequential_1_layer_call_fn_1614347�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614278
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614513
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614239
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614565�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_1613989�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *;�8
6�3
conv2d_10803_input�����������
�2�
.__inference_conv2d_10803_layer_call_fn_1614643�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_1614634�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
4__inference_max_pooling2d_5403_layer_call_fn_1614001�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_1613995�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
.__inference_conv2d_10804_layer_call_fn_1614663�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_1614654�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
4__inference_max_pooling2d_5404_layer_call_fn_1614013�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_1614007�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
.__inference_conv2d_10805_layer_call_fn_1614683�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_1614674�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
4__inference_max_pooling2d_5405_layer_call_fn_1614025�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_1614019�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
.__inference_flatten_1801_layer_call_fn_1614694�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
I__inference_flatten_1801_layer_call_and_return_conditional_losses_1614689�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
,__inference_dense_6303_layer_call_fn_1614714�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
G__inference_dense_6303_layer_call_and_return_conditional_losses_1614705�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
,__inference_dense_6304_layer_call_fn_1614734�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
G__inference_dense_6304_layer_call_and_return_conditional_losses_1614725�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
+__inference_dropout_1_layer_call_fn_1614761
+__inference_dropout_1_layer_call_fn_1614756�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614746
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614751�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dense_6305_layer_call_fn_1614781�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
G__inference_dense_6305_layer_call_and_return_conditional_losses_1614772�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
%__inference_signature_wrapper_1614454conv2d_10803_input"�
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
 �
"__inference__wrapped_model_1613989�&'45:;DEE�B
;�8
6�3
conv2d_10803_input�����������
� "7�4
2

dense_6305$�!

dense_6305����������
I__inference_conv2d_10803_layer_call_and_return_conditional_losses_1614634p9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0����������� 
� �
.__inference_conv2d_10803_layer_call_fn_1614643c9�6
/�,
*�'
inputs�����������
� ""������������ �
I__inference_conv2d_10804_layer_call_and_return_conditional_losses_1614654l7�4
-�*
(�%
inputs���������@@ 
� "-�*
#� 
0���������@@@
� �
.__inference_conv2d_10804_layer_call_fn_1614663_7�4
-�*
(�%
inputs���������@@ 
� " ����������@@@�
I__inference_conv2d_10805_layer_call_and_return_conditional_losses_1614674m&'7�4
-�*
(�%
inputs���������  @
� ".�+
$�!
0���������  �
� �
.__inference_conv2d_10805_layer_call_fn_1614683`&'7�4
-�*
(�%
inputs���������  @
� "!����������  ��
G__inference_dense_6303_layer_call_and_return_conditional_losses_1614705]450�-
&�#
!�
inputs����������d
� "%�"
�
0���������x
� �
,__inference_dense_6303_layer_call_fn_1614714P450�-
&�#
!�
inputs����������d
� "����������x�
G__inference_dense_6304_layer_call_and_return_conditional_losses_1614725\:;/�,
%�"
 �
inputs���������x
� "%�"
�
0���������<
� 
,__inference_dense_6304_layer_call_fn_1614734O:;/�,
%�"
 �
inputs���������x
� "����������<�
G__inference_dense_6305_layer_call_and_return_conditional_losses_1614772\DE/�,
%�"
 �
inputs���������<
� "%�"
�
0���������
� 
,__inference_dense_6305_layer_call_fn_1614781ODE/�,
%�"
 �
inputs���������<
� "�����������
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614746\3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� �
F__inference_dropout_1_layer_call_and_return_conditional_losses_1614751\3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� ~
+__inference_dropout_1_layer_call_fn_1614756O3�0
)�&
 �
inputs���������<
p
� "����������<~
+__inference_dropout_1_layer_call_fn_1614761O3�0
)�&
 �
inputs���������<
p 
� "����������<�
I__inference_flatten_1801_layer_call_and_return_conditional_losses_1614689b8�5
.�+
)�&
inputs���������

�
� "&�#
�
0����������d
� �
.__inference_flatten_1801_layer_call_fn_1614694U8�5
.�+
)�&
inputs���������

�
� "�����������d�
O__inference_max_pooling2d_5403_layer_call_and_return_conditional_losses_1613995�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
4__inference_max_pooling2d_5403_layer_call_fn_1614001�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
O__inference_max_pooling2d_5404_layer_call_and_return_conditional_losses_1614007�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
4__inference_max_pooling2d_5404_layer_call_fn_1614013�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
O__inference_max_pooling2d_5405_layer_call_and_return_conditional_losses_1614019�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
4__inference_max_pooling2d_5405_layer_call_fn_1614025�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614239�&'45:;DEM�J
C�@
6�3
conv2d_10803_input�����������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614278�&'45:;DEM�J
C�@
6�3
conv2d_10803_input�����������
p 

 
� "%�"
�
0���������
� �
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614513x&'45:;DEA�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_1_layer_call_and_return_conditional_losses_1614565x&'45:;DEA�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
.__inference_sequential_1_layer_call_fn_1614347w&'45:;DEM�J
C�@
6�3
conv2d_10803_input�����������
p

 
� "�����������
.__inference_sequential_1_layer_call_fn_1614415w&'45:;DEM�J
C�@
6�3
conv2d_10803_input�����������
p 

 
� "�����������
.__inference_sequential_1_layer_call_fn_1614594k&'45:;DEA�>
7�4
*�'
inputs�����������
p

 
� "�����������
.__inference_sequential_1_layer_call_fn_1614623k&'45:;DEA�>
7�4
*�'
inputs�����������
p 

 
� "�����������
%__inference_signature_wrapper_1614454�&'45:;DE[�X
� 
Q�N
L
conv2d_10803_input6�3
conv2d_10803_input�����������"7�4
2

dense_6305$�!

dense_6305���������