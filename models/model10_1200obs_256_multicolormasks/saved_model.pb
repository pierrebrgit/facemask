б
шО
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8См	

conv2d_65109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv2d_65109/kernel

'conv2d_65109/kernel/Read/ReadVariableOpReadVariableOpconv2d_65109/kernel*&
_output_shapes
: *
dtype0
z
conv2d_65109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_65109/bias
s
%conv2d_65109/bias/Read/ReadVariableOpReadVariableOpconv2d_65109/bias*
_output_shapes
: *
dtype0

conv2d_65110/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameconv2d_65110/kernel

'conv2d_65110/kernel/Read/ReadVariableOpReadVariableOpconv2d_65110/kernel*&
_output_shapes
: @*
dtype0
z
conv2d_65110/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_65110/bias
s
%conv2d_65110/bias/Read/ReadVariableOpReadVariableOpconv2d_65110/bias*
_output_shapes
:@*
dtype0

conv2d_65111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameconv2d_65111/kernel

'conv2d_65111/kernel/Read/ReadVariableOpReadVariableOpconv2d_65111/kernel*'
_output_shapes
:@*
dtype0
{
conv2d_65111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_65111/bias
t
%conv2d_65111/bias/Read/ReadVariableOpReadVariableOpconv2d_65111/bias*
_output_shapes	
:*
dtype0

dense_37989/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Йx*#
shared_namedense_37989/kernel
{
&dense_37989/kernel/Read/ReadVariableOpReadVariableOpdense_37989/kernel* 
_output_shapes
:
Йx*
dtype0
x
dense_37989/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*!
shared_namedense_37989/bias
q
$dense_37989/bias/Read/ReadVariableOpReadVariableOpdense_37989/bias*
_output_shapes
:x*
dtype0

dense_37990/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*#
shared_namedense_37990/kernel
y
&dense_37990/kernel/Read/ReadVariableOpReadVariableOpdense_37990/kernel*
_output_shapes

:x<*
dtype0
x
dense_37990/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_namedense_37990/bias
q
$dense_37990/bias/Read/ReadVariableOpReadVariableOpdense_37990/bias*
_output_shapes
:<*
dtype0

dense_37991/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*#
shared_namedense_37991/kernel
y
&dense_37991/kernel/Read/ReadVariableOpReadVariableOpdense_37991/kernel*
_output_shapes

:<*
dtype0
x
dense_37991/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_37991/bias
q
$dense_37991/bias/Read/ReadVariableOpReadVariableOpdense_37991/bias*
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

Adam/conv2d_65109/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv2d_65109/kernel/m

.Adam/conv2d_65109/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65109/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_65109/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_65109/bias/m

,Adam/conv2d_65109/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65109/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_65110/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv2d_65110/kernel/m

.Adam/conv2d_65110/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65110/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_65110/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_65110/bias/m

,Adam/conv2d_65110/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65110/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_65111/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/conv2d_65111/kernel/m

.Adam/conv2d_65111/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65111/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_65111/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_65111/bias/m

,Adam/conv2d_65111/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65111/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_37989/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Йx**
shared_nameAdam/dense_37989/kernel/m

-Adam/dense_37989/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37989/kernel/m* 
_output_shapes
:
Йx*
dtype0

Adam/dense_37989/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*(
shared_nameAdam/dense_37989/bias/m

+Adam/dense_37989/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37989/bias/m*
_output_shapes
:x*
dtype0

Adam/dense_37990/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<**
shared_nameAdam/dense_37990/kernel/m

-Adam/dense_37990/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37990/kernel/m*
_output_shapes

:x<*
dtype0

Adam/dense_37990/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dense_37990/bias/m

+Adam/dense_37990/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37990/bias/m*
_output_shapes
:<*
dtype0

Adam/dense_37991/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<**
shared_nameAdam/dense_37991/kernel/m

-Adam/dense_37991/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37991/kernel/m*
_output_shapes

:<*
dtype0

Adam/dense_37991/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_37991/bias/m

+Adam/dense_37991/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37991/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_65109/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv2d_65109/kernel/v

.Adam/conv2d_65109/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65109/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_65109/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_65109/bias/v

,Adam/conv2d_65109/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65109/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_65110/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv2d_65110/kernel/v

.Adam/conv2d_65110/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65110/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_65110/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_65110/bias/v

,Adam/conv2d_65110/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65110/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_65111/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/conv2d_65111/kernel/v

.Adam/conv2d_65111/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65111/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_65111/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_65111/bias/v

,Adam/conv2d_65111/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65111/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_37989/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Йx**
shared_nameAdam/dense_37989/kernel/v

-Adam/dense_37989/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37989/kernel/v* 
_output_shapes
:
Йx*
dtype0

Adam/dense_37989/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*(
shared_nameAdam/dense_37989/bias/v

+Adam/dense_37989/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37989/bias/v*
_output_shapes
:x*
dtype0

Adam/dense_37990/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<**
shared_nameAdam/dense_37990/kernel/v

-Adam/dense_37990/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37990/kernel/v*
_output_shapes

:x<*
dtype0

Adam/dense_37990/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dense_37990/bias/v

+Adam/dense_37990/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37990/bias/v*
_output_shapes
:<*
dtype0

Adam/dense_37991/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<**
shared_nameAdam/dense_37991/kernel/v

-Adam/dense_37991/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37991/kernel/v*
_output_shapes

:<*
dtype0

Adam/dense_37991/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_37991/bias/v

+Adam/dense_37991/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37991/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
дL
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*L
valueLBL BћK

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
А
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemmmm&m'm4m5m:m;mDm EmЁvЂvЃvЄvЅ&vІ'vЇ4vЈ5vЉ:vЊ;vЋDvЌEv­
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
­
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
VARIABLE_VALUEconv2d_65109/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_65109/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
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
­
Ymetrics

Zlayers
regularization_losses
[layer_regularization_losses
	variables
trainable_variables
\non_trainable_variables
]layer_metrics
_]
VARIABLE_VALUEconv2d_65110/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_65110/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
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
­
cmetrics

dlayers
"regularization_losses
elayer_regularization_losses
#	variables
$trainable_variables
fnon_trainable_variables
glayer_metrics
_]
VARIABLE_VALUEconv2d_65111/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_65111/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
­
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
­
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
­
rmetrics

slayers
0regularization_losses
tlayer_regularization_losses
1	variables
2trainable_variables
unon_trainable_variables
vlayer_metrics
^\
VARIABLE_VALUEdense_37989/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_37989/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
­
wmetrics

xlayers
6regularization_losses
ylayer_regularization_losses
7	variables
8trainable_variables
znon_trainable_variables
{layer_metrics
^\
VARIABLE_VALUEdense_37990/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_37990/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
Ў
|metrics

}layers
<regularization_losses
~layer_regularization_losses
=	variables
>trainable_variables
non_trainable_variables
layer_metrics
 
 
 
В
metrics
layers
@regularization_losses
 layer_regularization_losses
A	variables
Btrainable_variables
non_trainable_variables
layer_metrics
^\
VARIABLE_VALUEdense_37991/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_37991/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
В
metrics
layers
Fregularization_losses
 layer_regularization_losses
G	variables
Htrainable_variables
non_trainable_variables
layer_metrics
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
0
1
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

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

VARIABLE_VALUEAdam/conv2d_65109/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_65109/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_65110/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_65110/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_65111/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_65111/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_37989/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_37989/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_37990/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_37990/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_37991/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_37991/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_65109/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_65109/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_65110/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_65110/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_65111/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_65111/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_37989/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_37989/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_37990/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_37990/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_37991/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_37991/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

"serving_default_conv2d_65109_inputPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
У
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conv2d_65109_inputconv2d_65109/kernelconv2d_65109/biasconv2d_65110/kernelconv2d_65110/biasconv2d_65111/kernelconv2d_65111/biasdense_37989/kerneldense_37989/biasdense_37990/kerneldense_37990/biasdense_37991/kerneldense_37991/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_9074052
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'conv2d_65109/kernel/Read/ReadVariableOp%conv2d_65109/bias/Read/ReadVariableOp'conv2d_65110/kernel/Read/ReadVariableOp%conv2d_65110/bias/Read/ReadVariableOp'conv2d_65111/kernel/Read/ReadVariableOp%conv2d_65111/bias/Read/ReadVariableOp&dense_37989/kernel/Read/ReadVariableOp$dense_37989/bias/Read/ReadVariableOp&dense_37990/kernel/Read/ReadVariableOp$dense_37990/bias/Read/ReadVariableOp&dense_37991/kernel/Read/ReadVariableOp$dense_37991/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/conv2d_65109/kernel/m/Read/ReadVariableOp,Adam/conv2d_65109/bias/m/Read/ReadVariableOp.Adam/conv2d_65110/kernel/m/Read/ReadVariableOp,Adam/conv2d_65110/bias/m/Read/ReadVariableOp.Adam/conv2d_65111/kernel/m/Read/ReadVariableOp,Adam/conv2d_65111/bias/m/Read/ReadVariableOp-Adam/dense_37989/kernel/m/Read/ReadVariableOp+Adam/dense_37989/bias/m/Read/ReadVariableOp-Adam/dense_37990/kernel/m/Read/ReadVariableOp+Adam/dense_37990/bias/m/Read/ReadVariableOp-Adam/dense_37991/kernel/m/Read/ReadVariableOp+Adam/dense_37991/bias/m/Read/ReadVariableOp.Adam/conv2d_65109/kernel/v/Read/ReadVariableOp,Adam/conv2d_65109/bias/v/Read/ReadVariableOp.Adam/conv2d_65110/kernel/v/Read/ReadVariableOp,Adam/conv2d_65110/bias/v/Read/ReadVariableOp.Adam/conv2d_65111/kernel/v/Read/ReadVariableOp,Adam/conv2d_65111/bias/v/Read/ReadVariableOp-Adam/dense_37989/kernel/v/Read/ReadVariableOp+Adam/dense_37989/bias/v/Read/ReadVariableOp-Adam/dense_37990/kernel/v/Read/ReadVariableOp+Adam/dense_37990/bias/v/Read/ReadVariableOp-Adam/dense_37991/kernel/v/Read/ReadVariableOp+Adam/dense_37991/bias/v/Read/ReadVariableOpConst*:
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_9074537


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_65109/kernelconv2d_65109/biasconv2d_65110/kernelconv2d_65110/biasconv2d_65111/kernelconv2d_65111/biasdense_37989/kerneldense_37989/biasdense_37990/kerneldense_37990/biasdense_37991/kerneldense_37991/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_65109/kernel/mAdam/conv2d_65109/bias/mAdam/conv2d_65110/kernel/mAdam/conv2d_65110/bias/mAdam/conv2d_65111/kernel/mAdam/conv2d_65111/bias/mAdam/dense_37989/kernel/mAdam/dense_37989/bias/mAdam/dense_37990/kernel/mAdam/dense_37990/bias/mAdam/dense_37991/kernel/mAdam/dense_37991/bias/mAdam/conv2d_65109/kernel/vAdam/conv2d_65109/bias/vAdam/conv2d_65110/kernel/vAdam/conv2d_65110/bias/vAdam/conv2d_65111/kernel/vAdam/conv2d_65111/bias/vAdam/dense_37989/kernel/vAdam/dense_37989/bias/vAdam/dense_37990/kernel/vAdam/dense_37990/bias/vAdam/dense_37991/kernel/vAdam/dense_37991/bias/v*9
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_9074682
К
Q
5__inference_max_pooling2d_32567_layer_call_fn_9073623

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_90736172
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь

-__inference_dense_37989_layer_call_fn_9074312

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37989_layer_call_and_return_conditional_losses_90737362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџx2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџЙ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџЙ
 
_user_specified_nameinputs


.__inference_conv2d_65110_layer_call_fn_9074261

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_90736662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
п

т
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_9074232

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј	
с
H__inference_dense_37989_layer_call_and_return_conditional_losses_9073736

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџx2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџЙ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџЙ
 
_user_specified_nameinputs
ђ	
с
H__inference_dense_37990_layer_call_and_return_conditional_losses_9073763

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
H
№
I__inference_sequential_7_layer_call_and_return_conditional_losses_9074163

inputs/
+conv2d_65109_conv2d_readvariableop_resource0
,conv2d_65109_biasadd_readvariableop_resource/
+conv2d_65110_conv2d_readvariableop_resource0
,conv2d_65110_biasadd_readvariableop_resource/
+conv2d_65111_conv2d_readvariableop_resource0
,conv2d_65111_biasadd_readvariableop_resource.
*dense_37989_matmul_readvariableop_resource/
+dense_37989_biasadd_readvariableop_resource.
*dense_37990_matmul_readvariableop_resource/
+dense_37990_biasadd_readvariableop_resource.
*dense_37991_matmul_readvariableop_resource/
+dense_37991_biasadd_readvariableop_resource
identityЂ#conv2d_65109/BiasAdd/ReadVariableOpЂ"conv2d_65109/Conv2D/ReadVariableOpЂ#conv2d_65110/BiasAdd/ReadVariableOpЂ"conv2d_65110/Conv2D/ReadVariableOpЂ#conv2d_65111/BiasAdd/ReadVariableOpЂ"conv2d_65111/Conv2D/ReadVariableOpЂ"dense_37989/BiasAdd/ReadVariableOpЂ!dense_37989/MatMul/ReadVariableOpЂ"dense_37990/BiasAdd/ReadVariableOpЂ!dense_37990/MatMul/ReadVariableOpЂ"dense_37991/BiasAdd/ReadVariableOpЂ!dense_37991/MatMul/ReadVariableOpМ
"conv2d_65109/Conv2D/ReadVariableOpReadVariableOp+conv2d_65109_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"conv2d_65109/Conv2D/ReadVariableOpЬ
conv2d_65109/Conv2DConv2Dinputs*conv2d_65109/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d_65109/Conv2DГ
#conv2d_65109/BiasAdd/ReadVariableOpReadVariableOp,conv2d_65109_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv2d_65109/BiasAdd/ReadVariableOpО
conv2d_65109/BiasAddBiasAddconv2d_65109/Conv2D:output:0+conv2d_65109/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_65109/BiasAdd
conv2d_65109/ReluReluconv2d_65109/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_65109/Reluе
max_pooling2d_32565/MaxPoolMaxPoolconv2d_65109/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_32565/MaxPoolМ
"conv2d_65110/Conv2D/ReadVariableOpReadVariableOp+conv2d_65110_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"conv2d_65110/Conv2D/ReadVariableOpъ
conv2d_65110/Conv2DConv2D$max_pooling2d_32565/MaxPool:output:0*conv2d_65110/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_65110/Conv2DГ
#conv2d_65110/BiasAdd/ReadVariableOpReadVariableOp,conv2d_65110_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv2d_65110/BiasAdd/ReadVariableOpО
conv2d_65110/BiasAddBiasAddconv2d_65110/Conv2D:output:0+conv2d_65110/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_65110/BiasAdd
conv2d_65110/ReluReluconv2d_65110/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_65110/Reluг
max_pooling2d_32566/MaxPoolMaxPoolconv2d_65110/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_32566/MaxPoolН
"conv2d_65111/Conv2D/ReadVariableOpReadVariableOp+conv2d_65111_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"conv2d_65111/Conv2D/ReadVariableOpщ
conv2d_65111/Conv2DConv2D$max_pooling2d_32566/MaxPool:output:0*conv2d_65111/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_65111/Conv2DД
#conv2d_65111/BiasAdd/ReadVariableOpReadVariableOp,conv2d_65111_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#conv2d_65111/BiasAdd/ReadVariableOpН
conv2d_65111/BiasAddBiasAddconv2d_65111/Conv2D:output:0+conv2d_65111/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_65111/BiasAdd
conv2d_65111/ReluReluconv2d_65111/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_65111/Reluд
max_pooling2d_32567/MaxPoolMaxPoolconv2d_65111/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_32567/MaxPool{
flatten_10855/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2
flatten_10855/ConstБ
flatten_10855/ReshapeReshape$max_pooling2d_32567/MaxPool:output:0flatten_10855/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2
flatten_10855/ReshapeГ
!dense_37989/MatMul/ReadVariableOpReadVariableOp*dense_37989_matmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype02#
!dense_37989/MatMul/ReadVariableOpЏ
dense_37989/MatMulMatMulflatten_10855/Reshape:output:0)dense_37989/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_37989/MatMulА
"dense_37989/BiasAdd/ReadVariableOpReadVariableOp+dense_37989_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02$
"dense_37989/BiasAdd/ReadVariableOpБ
dense_37989/BiasAddBiasAdddense_37989/MatMul:product:0*dense_37989/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_37989/BiasAdd|
dense_37989/ReluReludense_37989/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_37989/ReluБ
!dense_37990/MatMul/ReadVariableOpReadVariableOp*dense_37990_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02#
!dense_37990/MatMul/ReadVariableOpЏ
dense_37990/MatMulMatMuldense_37989/Relu:activations:0)dense_37990/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_37990/MatMulА
"dense_37990/BiasAdd/ReadVariableOpReadVariableOp+dense_37990_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dense_37990/BiasAdd/ReadVariableOpБ
dense_37990/BiasAddBiasAdddense_37990/MatMul:product:0*dense_37990/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_37990/BiasAdd|
dense_37990/ReluReludense_37990/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_37990/Relu
dropout_7/IdentityIdentitydense_37990/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout_7/IdentityБ
!dense_37991/MatMul/ReadVariableOpReadVariableOp*dense_37991_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!dense_37991/MatMul/ReadVariableOpЌ
dense_37991/MatMulMatMuldropout_7/Identity:output:0)dense_37991/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_37991/MatMulА
"dense_37991/BiasAdd/ReadVariableOpReadVariableOp+dense_37991_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_37991/BiasAdd/ReadVariableOpБ
dense_37991/BiasAddBiasAdddense_37991/MatMul:product:0*dense_37991/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_37991/BiasAdd
dense_37991/SoftmaxSoftmaxdense_37991/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_37991/Softmax­
IdentityIdentitydense_37991/Softmax:softmax:0$^conv2d_65109/BiasAdd/ReadVariableOp#^conv2d_65109/Conv2D/ReadVariableOp$^conv2d_65110/BiasAdd/ReadVariableOp#^conv2d_65110/Conv2D/ReadVariableOp$^conv2d_65111/BiasAdd/ReadVariableOp#^conv2d_65111/Conv2D/ReadVariableOp#^dense_37989/BiasAdd/ReadVariableOp"^dense_37989/MatMul/ReadVariableOp#^dense_37990/BiasAdd/ReadVariableOp"^dense_37990/MatMul/ReadVariableOp#^dense_37991/BiasAdd/ReadVariableOp"^dense_37991/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2J
#conv2d_65109/BiasAdd/ReadVariableOp#conv2d_65109/BiasAdd/ReadVariableOp2H
"conv2d_65109/Conv2D/ReadVariableOp"conv2d_65109/Conv2D/ReadVariableOp2J
#conv2d_65110/BiasAdd/ReadVariableOp#conv2d_65110/BiasAdd/ReadVariableOp2H
"conv2d_65110/Conv2D/ReadVariableOp"conv2d_65110/Conv2D/ReadVariableOp2J
#conv2d_65111/BiasAdd/ReadVariableOp#conv2d_65111/BiasAdd/ReadVariableOp2H
"conv2d_65111/Conv2D/ReadVariableOp"conv2d_65111/Conv2D/ReadVariableOp2H
"dense_37989/BiasAdd/ReadVariableOp"dense_37989/BiasAdd/ReadVariableOp2F
!dense_37989/MatMul/ReadVariableOp!dense_37989/MatMul/ReadVariableOp2H
"dense_37990/BiasAdd/ReadVariableOp"dense_37990/BiasAdd/ReadVariableOp2F
!dense_37990/MatMul/ReadVariableOp!dense_37990/MatMul/ReadVariableOp2H
"dense_37991/BiasAdd/ReadVariableOp"dense_37991/BiasAdd/ReadVariableOp2F
!dense_37991/MatMul/ReadVariableOp!dense_37991/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
d
+__inference_dropout_7_layer_call_fn_9074354

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_90737912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ<22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
И3

I__inference_sequential_7_layer_call_and_return_conditional_losses_9073876
conv2d_65109_input
conv2d_65109_9073840
conv2d_65109_9073842
conv2d_65110_9073846
conv2d_65110_9073848
conv2d_65111_9073852
conv2d_65111_9073854
dense_37989_9073859
dense_37989_9073861
dense_37990_9073864
dense_37990_9073866
dense_37991_9073870
dense_37991_9073872
identityЂ$conv2d_65109/StatefulPartitionedCallЂ$conv2d_65110/StatefulPartitionedCallЂ$conv2d_65111/StatefulPartitionedCallЂ#dense_37989/StatefulPartitionedCallЂ#dense_37990/StatefulPartitionedCallЂ#dense_37991/StatefulPartitionedCallФ
$conv2d_65109/StatefulPartitionedCallStatefulPartitionedCallconv2d_65109_inputconv2d_65109_9073840conv2d_65109_9073842*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_90736382&
$conv2d_65109/StatefulPartitionedCallЊ
#max_pooling2d_32565/PartitionedCallPartitionedCall-conv2d_65109/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_90735932%
#max_pooling2d_32565/PartitionedCallо
$conv2d_65110/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32565/PartitionedCall:output:0conv2d_65110_9073846conv2d_65110_9073848*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_90736662&
$conv2d_65110/StatefulPartitionedCallЈ
#max_pooling2d_32566/PartitionedCallPartitionedCall-conv2d_65110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_90736052%
#max_pooling2d_32566/PartitionedCallн
$conv2d_65111/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32566/PartitionedCall:output:0conv2d_65111_9073852conv2d_65111_9073854*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_90736942&
$conv2d_65111/StatefulPartitionedCallЉ
#max_pooling2d_32567/PartitionedCallPartitionedCall-conv2d_65111/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_90736172%
#max_pooling2d_32567/PartitionedCall
flatten_10855/PartitionedCallPartitionedCall,max_pooling2d_32567/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџЙ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_flatten_10855_layer_call_and_return_conditional_losses_90737172
flatten_10855/PartitionedCallЩ
#dense_37989/StatefulPartitionedCallStatefulPartitionedCall&flatten_10855/PartitionedCall:output:0dense_37989_9073859dense_37989_9073861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37989_layer_call_and_return_conditional_losses_90737362%
#dense_37989/StatefulPartitionedCallЯ
#dense_37990/StatefulPartitionedCallStatefulPartitionedCall,dense_37989/StatefulPartitionedCall:output:0dense_37990_9073864dense_37990_9073866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37990_layer_call_and_return_conditional_losses_90737632%
#dense_37990/StatefulPartitionedCall
dropout_7/PartitionedCallPartitionedCall,dense_37990/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_90737962
dropout_7/PartitionedCallХ
#dense_37991/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_37991_9073870dense_37991_9073872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37991_layer_call_and_return_conditional_losses_90738202%
#dense_37991/StatefulPartitionedCallч
IdentityIdentity,dense_37991/StatefulPartitionedCall:output:0%^conv2d_65109/StatefulPartitionedCall%^conv2d_65110/StatefulPartitionedCall%^conv2d_65111/StatefulPartitionedCall$^dense_37989/StatefulPartitionedCall$^dense_37990/StatefulPartitionedCall$^dense_37991/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_65109/StatefulPartitionedCall$conv2d_65109/StatefulPartitionedCall2L
$conv2d_65110/StatefulPartitionedCall$conv2d_65110/StatefulPartitionedCall2L
$conv2d_65111/StatefulPartitionedCall$conv2d_65111/StatefulPartitionedCall2J
#dense_37989/StatefulPartitionedCall#dense_37989/StatefulPartitionedCall2J
#dense_37990/StatefulPartitionedCall#dense_37990/StatefulPartitionedCall2J
#dense_37991/StatefulPartitionedCall#dense_37991/StatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_65109_input
Н_
Ћ
 __inference__traced_save_9074537
file_prefix2
.savev2_conv2d_65109_kernel_read_readvariableop0
,savev2_conv2d_65109_bias_read_readvariableop2
.savev2_conv2d_65110_kernel_read_readvariableop0
,savev2_conv2d_65110_bias_read_readvariableop2
.savev2_conv2d_65111_kernel_read_readvariableop0
,savev2_conv2d_65111_bias_read_readvariableop1
-savev2_dense_37989_kernel_read_readvariableop/
+savev2_dense_37989_bias_read_readvariableop1
-savev2_dense_37990_kernel_read_readvariableop/
+savev2_dense_37990_bias_read_readvariableop1
-savev2_dense_37991_kernel_read_readvariableop/
+savev2_dense_37991_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_conv2d_65109_kernel_m_read_readvariableop7
3savev2_adam_conv2d_65109_bias_m_read_readvariableop9
5savev2_adam_conv2d_65110_kernel_m_read_readvariableop7
3savev2_adam_conv2d_65110_bias_m_read_readvariableop9
5savev2_adam_conv2d_65111_kernel_m_read_readvariableop7
3savev2_adam_conv2d_65111_bias_m_read_readvariableop8
4savev2_adam_dense_37989_kernel_m_read_readvariableop6
2savev2_adam_dense_37989_bias_m_read_readvariableop8
4savev2_adam_dense_37990_kernel_m_read_readvariableop6
2savev2_adam_dense_37990_bias_m_read_readvariableop8
4savev2_adam_dense_37991_kernel_m_read_readvariableop6
2savev2_adam_dense_37991_bias_m_read_readvariableop9
5savev2_adam_conv2d_65109_kernel_v_read_readvariableop7
3savev2_adam_conv2d_65109_bias_v_read_readvariableop9
5savev2_adam_conv2d_65110_kernel_v_read_readvariableop7
3savev2_adam_conv2d_65110_bias_v_read_readvariableop9
5savev2_adam_conv2d_65111_kernel_v_read_readvariableop7
3savev2_adam_conv2d_65111_bias_v_read_readvariableop8
4savev2_adam_dense_37989_kernel_v_read_readvariableop6
2savev2_adam_dense_37989_bias_v_read_readvariableop8
4savev2_adam_dense_37990_kernel_v_read_readvariableop6
2savev2_adam_dense_37990_bias_v_read_readvariableop8
4savev2_adam_dense_37991_kernel_v_read_readvariableop6
2savev2_adam_dense_37991_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameК
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv2d_65109_kernel_read_readvariableop,savev2_conv2d_65109_bias_read_readvariableop.savev2_conv2d_65110_kernel_read_readvariableop,savev2_conv2d_65110_bias_read_readvariableop.savev2_conv2d_65111_kernel_read_readvariableop,savev2_conv2d_65111_bias_read_readvariableop-savev2_dense_37989_kernel_read_readvariableop+savev2_dense_37989_bias_read_readvariableop-savev2_dense_37990_kernel_read_readvariableop+savev2_dense_37990_bias_read_readvariableop-savev2_dense_37991_kernel_read_readvariableop+savev2_dense_37991_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_conv2d_65109_kernel_m_read_readvariableop3savev2_adam_conv2d_65109_bias_m_read_readvariableop5savev2_adam_conv2d_65110_kernel_m_read_readvariableop3savev2_adam_conv2d_65110_bias_m_read_readvariableop5savev2_adam_conv2d_65111_kernel_m_read_readvariableop3savev2_adam_conv2d_65111_bias_m_read_readvariableop4savev2_adam_dense_37989_kernel_m_read_readvariableop2savev2_adam_dense_37989_bias_m_read_readvariableop4savev2_adam_dense_37990_kernel_m_read_readvariableop2savev2_adam_dense_37990_bias_m_read_readvariableop4savev2_adam_dense_37991_kernel_m_read_readvariableop2savev2_adam_dense_37991_bias_m_read_readvariableop5savev2_adam_conv2d_65109_kernel_v_read_readvariableop3savev2_adam_conv2d_65109_bias_v_read_readvariableop5savev2_adam_conv2d_65110_kernel_v_read_readvariableop3savev2_adam_conv2d_65110_bias_v_read_readvariableop5savev2_adam_conv2d_65111_kernel_v_read_readvariableop3savev2_adam_conv2d_65111_bias_v_read_readvariableop4savev2_adam_dense_37989_kernel_v_read_readvariableop2savev2_adam_dense_37989_bias_v_read_readvariableop4savev2_adam_dense_37990_kernel_v_read_readvariableop2savev2_adam_dense_37990_bias_v_read_readvariableop4savev2_adam_dense_37991_kernel_v_read_readvariableop2savev2_adam_dense_37991_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*
_input_shapes
: : : : @:@:@::
Йx:x:x<:<:<:: : : : : : : : : : : : @:@:@::
Йx:x:x<:<:<:: : : @:@:@::
Йx:x:x<:<:<:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 
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
:@:!

_output_shapes	
::&"
 
_output_shapes
:
Йx: 
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
: : 
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
:@:!

_output_shapes	
::&"
 
_output_shapes
:
Йx: 
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
: : #
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
:@:!'

_output_shapes	
::&("
 
_output_shapes
:
Йx: )
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
ј	
с
H__inference_dense_37989_layer_call_and_return_conditional_losses_9074303

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџx2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџЙ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџЙ
 
_user_specified_nameinputs
ВZ

"__inference__wrapped_model_9073587
conv2d_65109_input<
8sequential_7_conv2d_65109_conv2d_readvariableop_resource=
9sequential_7_conv2d_65109_biasadd_readvariableop_resource<
8sequential_7_conv2d_65110_conv2d_readvariableop_resource=
9sequential_7_conv2d_65110_biasadd_readvariableop_resource<
8sequential_7_conv2d_65111_conv2d_readvariableop_resource=
9sequential_7_conv2d_65111_biasadd_readvariableop_resource;
7sequential_7_dense_37989_matmul_readvariableop_resource<
8sequential_7_dense_37989_biasadd_readvariableop_resource;
7sequential_7_dense_37990_matmul_readvariableop_resource<
8sequential_7_dense_37990_biasadd_readvariableop_resource;
7sequential_7_dense_37991_matmul_readvariableop_resource<
8sequential_7_dense_37991_biasadd_readvariableop_resource
identityЂ0sequential_7/conv2d_65109/BiasAdd/ReadVariableOpЂ/sequential_7/conv2d_65109/Conv2D/ReadVariableOpЂ0sequential_7/conv2d_65110/BiasAdd/ReadVariableOpЂ/sequential_7/conv2d_65110/Conv2D/ReadVariableOpЂ0sequential_7/conv2d_65111/BiasAdd/ReadVariableOpЂ/sequential_7/conv2d_65111/Conv2D/ReadVariableOpЂ/sequential_7/dense_37989/BiasAdd/ReadVariableOpЂ.sequential_7/dense_37989/MatMul/ReadVariableOpЂ/sequential_7/dense_37990/BiasAdd/ReadVariableOpЂ.sequential_7/dense_37990/MatMul/ReadVariableOpЂ/sequential_7/dense_37991/BiasAdd/ReadVariableOpЂ.sequential_7/dense_37991/MatMul/ReadVariableOpу
/sequential_7/conv2d_65109/Conv2D/ReadVariableOpReadVariableOp8sequential_7_conv2d_65109_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/sequential_7/conv2d_65109/Conv2D/ReadVariableOpџ
 sequential_7/conv2d_65109/Conv2DConv2Dconv2d_65109_input7sequential_7/conv2d_65109/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2"
 sequential_7/conv2d_65109/Conv2Dк
0sequential_7/conv2d_65109/BiasAdd/ReadVariableOpReadVariableOp9sequential_7_conv2d_65109_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential_7/conv2d_65109/BiasAdd/ReadVariableOpђ
!sequential_7/conv2d_65109/BiasAddBiasAdd)sequential_7/conv2d_65109/Conv2D:output:08sequential_7/conv2d_65109/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2#
!sequential_7/conv2d_65109/BiasAddА
sequential_7/conv2d_65109/ReluRelu*sequential_7/conv2d_65109/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2 
sequential_7/conv2d_65109/Reluќ
(sequential_7/max_pooling2d_32565/MaxPoolMaxPool,sequential_7/conv2d_65109/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2*
(sequential_7/max_pooling2d_32565/MaxPoolу
/sequential_7/conv2d_65110/Conv2D/ReadVariableOpReadVariableOp8sequential_7_conv2d_65110_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/sequential_7/conv2d_65110/Conv2D/ReadVariableOp
 sequential_7/conv2d_65110/Conv2DConv2D1sequential_7/max_pooling2d_32565/MaxPool:output:07sequential_7/conv2d_65110/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2"
 sequential_7/conv2d_65110/Conv2Dк
0sequential_7/conv2d_65110/BiasAdd/ReadVariableOpReadVariableOp9sequential_7_conv2d_65110_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential_7/conv2d_65110/BiasAdd/ReadVariableOpђ
!sequential_7/conv2d_65110/BiasAddBiasAdd)sequential_7/conv2d_65110/Conv2D:output:08sequential_7/conv2d_65110/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2#
!sequential_7/conv2d_65110/BiasAddА
sequential_7/conv2d_65110/ReluRelu*sequential_7/conv2d_65110/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2 
sequential_7/conv2d_65110/Reluњ
(sequential_7/max_pooling2d_32566/MaxPoolMaxPool,sequential_7/conv2d_65110/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@@*
ksize
*
paddingVALID*
strides
2*
(sequential_7/max_pooling2d_32566/MaxPoolф
/sequential_7/conv2d_65111/Conv2D/ReadVariableOpReadVariableOp8sequential_7_conv2d_65111_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype021
/sequential_7/conv2d_65111/Conv2D/ReadVariableOp
 sequential_7/conv2d_65111/Conv2DConv2D1sequential_7/max_pooling2d_32566/MaxPool:output:07sequential_7/conv2d_65111/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2"
 sequential_7/conv2d_65111/Conv2Dл
0sequential_7/conv2d_65111/BiasAdd/ReadVariableOpReadVariableOp9sequential_7_conv2d_65111_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0sequential_7/conv2d_65111/BiasAdd/ReadVariableOpё
!sequential_7/conv2d_65111/BiasAddBiasAdd)sequential_7/conv2d_65111/Conv2D:output:08sequential_7/conv2d_65111/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2#
!sequential_7/conv2d_65111/BiasAddЏ
sequential_7/conv2d_65111/ReluRelu*sequential_7/conv2d_65111/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2 
sequential_7/conv2d_65111/Reluћ
(sequential_7/max_pooling2d_32567/MaxPoolMaxPool,sequential_7/conv2d_65111/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2*
(sequential_7/max_pooling2d_32567/MaxPool
 sequential_7/flatten_10855/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2"
 sequential_7/flatten_10855/Constх
"sequential_7/flatten_10855/ReshapeReshape1sequential_7/max_pooling2d_32567/MaxPool:output:0)sequential_7/flatten_10855/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2$
"sequential_7/flatten_10855/Reshapeк
.sequential_7/dense_37989/MatMul/ReadVariableOpReadVariableOp7sequential_7_dense_37989_matmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype020
.sequential_7/dense_37989/MatMul/ReadVariableOpу
sequential_7/dense_37989/MatMulMatMul+sequential_7/flatten_10855/Reshape:output:06sequential_7/dense_37989/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2!
sequential_7/dense_37989/MatMulз
/sequential_7/dense_37989/BiasAdd/ReadVariableOpReadVariableOp8sequential_7_dense_37989_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype021
/sequential_7/dense_37989/BiasAdd/ReadVariableOpх
 sequential_7/dense_37989/BiasAddBiasAdd)sequential_7/dense_37989/MatMul:product:07sequential_7/dense_37989/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2"
 sequential_7/dense_37989/BiasAddЃ
sequential_7/dense_37989/ReluRelu)sequential_7/dense_37989/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
sequential_7/dense_37989/Reluи
.sequential_7/dense_37990/MatMul/ReadVariableOpReadVariableOp7sequential_7_dense_37990_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype020
.sequential_7/dense_37990/MatMul/ReadVariableOpу
sequential_7/dense_37990/MatMulMatMul+sequential_7/dense_37989/Relu:activations:06sequential_7/dense_37990/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
sequential_7/dense_37990/MatMulз
/sequential_7/dense_37990/BiasAdd/ReadVariableOpReadVariableOp8sequential_7_dense_37990_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype021
/sequential_7/dense_37990/BiasAdd/ReadVariableOpх
 sequential_7/dense_37990/BiasAddBiasAdd)sequential_7/dense_37990/MatMul:product:07sequential_7/dense_37990/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 sequential_7/dense_37990/BiasAddЃ
sequential_7/dense_37990/ReluRelu)sequential_7/dense_37990/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
sequential_7/dense_37990/Relu­
sequential_7/dropout_7/IdentityIdentity+sequential_7/dense_37990/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
sequential_7/dropout_7/Identityи
.sequential_7/dense_37991/MatMul/ReadVariableOpReadVariableOp7sequential_7_dense_37991_matmul_readvariableop_resource*
_output_shapes

:<*
dtype020
.sequential_7/dense_37991/MatMul/ReadVariableOpр
sequential_7/dense_37991/MatMulMatMul(sequential_7/dropout_7/Identity:output:06sequential_7/dense_37991/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_7/dense_37991/MatMulз
/sequential_7/dense_37991/BiasAdd/ReadVariableOpReadVariableOp8sequential_7_dense_37991_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_7/dense_37991/BiasAdd/ReadVariableOpх
 sequential_7/dense_37991/BiasAddBiasAdd)sequential_7/dense_37991/MatMul:product:07sequential_7/dense_37991/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_7/dense_37991/BiasAddЌ
 sequential_7/dense_37991/SoftmaxSoftmax)sequential_7/dense_37991/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_7/dense_37991/Softmaxж
IdentityIdentity*sequential_7/dense_37991/Softmax:softmax:01^sequential_7/conv2d_65109/BiasAdd/ReadVariableOp0^sequential_7/conv2d_65109/Conv2D/ReadVariableOp1^sequential_7/conv2d_65110/BiasAdd/ReadVariableOp0^sequential_7/conv2d_65110/Conv2D/ReadVariableOp1^sequential_7/conv2d_65111/BiasAdd/ReadVariableOp0^sequential_7/conv2d_65111/Conv2D/ReadVariableOp0^sequential_7/dense_37989/BiasAdd/ReadVariableOp/^sequential_7/dense_37989/MatMul/ReadVariableOp0^sequential_7/dense_37990/BiasAdd/ReadVariableOp/^sequential_7/dense_37990/MatMul/ReadVariableOp0^sequential_7/dense_37991/BiasAdd/ReadVariableOp/^sequential_7/dense_37991/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2d
0sequential_7/conv2d_65109/BiasAdd/ReadVariableOp0sequential_7/conv2d_65109/BiasAdd/ReadVariableOp2b
/sequential_7/conv2d_65109/Conv2D/ReadVariableOp/sequential_7/conv2d_65109/Conv2D/ReadVariableOp2d
0sequential_7/conv2d_65110/BiasAdd/ReadVariableOp0sequential_7/conv2d_65110/BiasAdd/ReadVariableOp2b
/sequential_7/conv2d_65110/Conv2D/ReadVariableOp/sequential_7/conv2d_65110/Conv2D/ReadVariableOp2d
0sequential_7/conv2d_65111/BiasAdd/ReadVariableOp0sequential_7/conv2d_65111/BiasAdd/ReadVariableOp2b
/sequential_7/conv2d_65111/Conv2D/ReadVariableOp/sequential_7/conv2d_65111/Conv2D/ReadVariableOp2b
/sequential_7/dense_37989/BiasAdd/ReadVariableOp/sequential_7/dense_37989/BiasAdd/ReadVariableOp2`
.sequential_7/dense_37989/MatMul/ReadVariableOp.sequential_7/dense_37989/MatMul/ReadVariableOp2b
/sequential_7/dense_37990/BiasAdd/ReadVariableOp/sequential_7/dense_37990/BiasAdd/ReadVariableOp2`
.sequential_7/dense_37990/MatMul/ReadVariableOp.sequential_7/dense_37990/MatMul/ReadVariableOp2b
/sequential_7/dense_37991/BiasAdd/ReadVariableOp/sequential_7/dense_37991/BiasAdd/ReadVariableOp2`
.sequential_7/dense_37991/MatMul/ReadVariableOp.sequential_7/dense_37991/MatMul/ReadVariableOp:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_65109_input
Ц
f
J__inference_flatten_10855_layer_call_and_return_conditional_losses_9074287

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й

т
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_9073694

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@@
 
_user_specified_nameinputs
п

т
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_9073666

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ђ	
с
H__inference_dense_37990_layer_call_and_return_conditional_losses_9074323

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
ш

-__inference_dense_37990_layer_call_fn_9074332

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37990_layer_call_and_return_conditional_losses_90737632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs

e
F__inference_dropout_7_layer_call_and_return_conditional_losses_9073791

inputs
identityc
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
:џџџџџџџџџ<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ<:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs


.__inference_conv2d_65109_layer_call_fn_9074241

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_90736382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ	
с
H__inference_dense_37991_layer_call_and_return_conditional_losses_9074370

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
ф	
І
.__inference_sequential_7_layer_call_fn_9073945
conv2d_65109_input
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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_65109_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_90739182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_65109_input
З
K
/__inference_flatten_10855_layer_call_fn_9074292

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџЙ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_flatten_10855_layer_call_and_return_conditional_losses_90737172
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


.__inference_conv2d_65111_layer_call_fn_9074281

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_90736942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@@
 
_user_specified_nameinputs
њ	
с
H__inference_dense_37991_layer_call_and_return_conditional_losses_9073820

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
Ф4
Ѕ
I__inference_sequential_7_layer_call_and_return_conditional_losses_9073918

inputs
conv2d_65109_9073882
conv2d_65109_9073884
conv2d_65110_9073888
conv2d_65110_9073890
conv2d_65111_9073894
conv2d_65111_9073896
dense_37989_9073901
dense_37989_9073903
dense_37990_9073906
dense_37990_9073908
dense_37991_9073912
dense_37991_9073914
identityЂ$conv2d_65109/StatefulPartitionedCallЂ$conv2d_65110/StatefulPartitionedCallЂ$conv2d_65111/StatefulPartitionedCallЂ#dense_37989/StatefulPartitionedCallЂ#dense_37990/StatefulPartitionedCallЂ#dense_37991/StatefulPartitionedCallЂ!dropout_7/StatefulPartitionedCallИ
$conv2d_65109/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_65109_9073882conv2d_65109_9073884*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_90736382&
$conv2d_65109/StatefulPartitionedCallЊ
#max_pooling2d_32565/PartitionedCallPartitionedCall-conv2d_65109/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_90735932%
#max_pooling2d_32565/PartitionedCallо
$conv2d_65110/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32565/PartitionedCall:output:0conv2d_65110_9073888conv2d_65110_9073890*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_90736662&
$conv2d_65110/StatefulPartitionedCallЈ
#max_pooling2d_32566/PartitionedCallPartitionedCall-conv2d_65110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_90736052%
#max_pooling2d_32566/PartitionedCallн
$conv2d_65111/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32566/PartitionedCall:output:0conv2d_65111_9073894conv2d_65111_9073896*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_90736942&
$conv2d_65111/StatefulPartitionedCallЉ
#max_pooling2d_32567/PartitionedCallPartitionedCall-conv2d_65111/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_90736172%
#max_pooling2d_32567/PartitionedCall
flatten_10855/PartitionedCallPartitionedCall,max_pooling2d_32567/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџЙ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_flatten_10855_layer_call_and_return_conditional_losses_90737172
flatten_10855/PartitionedCallЩ
#dense_37989/StatefulPartitionedCallStatefulPartitionedCall&flatten_10855/PartitionedCall:output:0dense_37989_9073901dense_37989_9073903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37989_layer_call_and_return_conditional_losses_90737362%
#dense_37989/StatefulPartitionedCallЯ
#dense_37990/StatefulPartitionedCallStatefulPartitionedCall,dense_37989/StatefulPartitionedCall:output:0dense_37990_9073906dense_37990_9073908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37990_layer_call_and_return_conditional_losses_90737632%
#dense_37990/StatefulPartitionedCall
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall,dense_37990/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_90737912#
!dropout_7/StatefulPartitionedCallЭ
#dense_37991/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_37991_9073912dense_37991_9073914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37991_layer_call_and_return_conditional_losses_90738202%
#dense_37991/StatefulPartitionedCall
IdentityIdentity,dense_37991/StatefulPartitionedCall:output:0%^conv2d_65109/StatefulPartitionedCall%^conv2d_65110/StatefulPartitionedCall%^conv2d_65111/StatefulPartitionedCall$^dense_37989/StatefulPartitionedCall$^dense_37990/StatefulPartitionedCall$^dense_37991/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_65109/StatefulPartitionedCall$conv2d_65109/StatefulPartitionedCall2L
$conv2d_65110/StatefulPartitionedCall$conv2d_65110/StatefulPartitionedCall2L
$conv2d_65111/StatefulPartitionedCall$conv2d_65111/StatefulPartitionedCall2J
#dense_37989/StatefulPartitionedCall#dense_37989/StatefulPartitionedCall2J
#dense_37990/StatefulPartitionedCall#dense_37990/StatefulPartitionedCall2J
#dense_37991/StatefulPartitionedCall#dense_37991/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_dropout_7_layer_call_fn_9074359

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_90737962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ<:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
ф	
І
.__inference_sequential_7_layer_call_fn_9074013
conv2d_65109_input
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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_65109_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_90739862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_65109_input
Р	

.__inference_sequential_7_layer_call_fn_9074221

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
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_90739862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
3

I__inference_sequential_7_layer_call_and_return_conditional_losses_9073986

inputs
conv2d_65109_9073950
conv2d_65109_9073952
conv2d_65110_9073956
conv2d_65110_9073958
conv2d_65111_9073962
conv2d_65111_9073964
dense_37989_9073969
dense_37989_9073971
dense_37990_9073974
dense_37990_9073976
dense_37991_9073980
dense_37991_9073982
identityЂ$conv2d_65109/StatefulPartitionedCallЂ$conv2d_65110/StatefulPartitionedCallЂ$conv2d_65111/StatefulPartitionedCallЂ#dense_37989/StatefulPartitionedCallЂ#dense_37990/StatefulPartitionedCallЂ#dense_37991/StatefulPartitionedCallИ
$conv2d_65109/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_65109_9073950conv2d_65109_9073952*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_90736382&
$conv2d_65109/StatefulPartitionedCallЊ
#max_pooling2d_32565/PartitionedCallPartitionedCall-conv2d_65109/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_90735932%
#max_pooling2d_32565/PartitionedCallо
$conv2d_65110/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32565/PartitionedCall:output:0conv2d_65110_9073956conv2d_65110_9073958*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_90736662&
$conv2d_65110/StatefulPartitionedCallЈ
#max_pooling2d_32566/PartitionedCallPartitionedCall-conv2d_65110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_90736052%
#max_pooling2d_32566/PartitionedCallн
$conv2d_65111/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32566/PartitionedCall:output:0conv2d_65111_9073962conv2d_65111_9073964*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_90736942&
$conv2d_65111/StatefulPartitionedCallЉ
#max_pooling2d_32567/PartitionedCallPartitionedCall-conv2d_65111/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_90736172%
#max_pooling2d_32567/PartitionedCall
flatten_10855/PartitionedCallPartitionedCall,max_pooling2d_32567/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџЙ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_flatten_10855_layer_call_and_return_conditional_losses_90737172
flatten_10855/PartitionedCallЩ
#dense_37989/StatefulPartitionedCallStatefulPartitionedCall&flatten_10855/PartitionedCall:output:0dense_37989_9073969dense_37989_9073971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37989_layer_call_and_return_conditional_losses_90737362%
#dense_37989/StatefulPartitionedCallЯ
#dense_37990/StatefulPartitionedCallStatefulPartitionedCall,dense_37989/StatefulPartitionedCall:output:0dense_37990_9073974dense_37990_9073976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37990_layer_call_and_return_conditional_losses_90737632%
#dense_37990/StatefulPartitionedCall
dropout_7/PartitionedCallPartitionedCall,dense_37990/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_90737962
dropout_7/PartitionedCallХ
#dense_37991/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_37991_9073980dense_37991_9073982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37991_layer_call_and_return_conditional_losses_90738202%
#dense_37991/StatefulPartitionedCallч
IdentityIdentity,dense_37991/StatefulPartitionedCall:output:0%^conv2d_65109/StatefulPartitionedCall%^conv2d_65110/StatefulPartitionedCall%^conv2d_65111/StatefulPartitionedCall$^dense_37989/StatefulPartitionedCall$^dense_37990/StatefulPartitionedCall$^dense_37991/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_65109/StatefulPartitionedCall$conv2d_65109/StatefulPartitionedCall2L
$conv2d_65110/StatefulPartitionedCall$conv2d_65110/StatefulPartitionedCall2L
$conv2d_65111/StatefulPartitionedCall$conv2d_65111/StatefulPartitionedCall2J
#dense_37989/StatefulPartitionedCall#dense_37989/StatefulPartitionedCall2J
#dense_37990/StatefulPartitionedCall#dense_37990/StatefulPartitionedCall2J
#dense_37991/StatefulPartitionedCall#dense_37991/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
Q
5__inference_max_pooling2d_32565_layer_call_fn_9073599

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_90735932
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д	

%__inference_signature_wrapper_9074052
conv2d_65109_input
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
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallconv2d_65109_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_90735872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_65109_input
Р	

.__inference_sequential_7_layer_call_fn_9074192

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
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_90739182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш4
Б
I__inference_sequential_7_layer_call_and_return_conditional_losses_9073837
conv2d_65109_input
conv2d_65109_9073649
conv2d_65109_9073651
conv2d_65110_9073677
conv2d_65110_9073679
conv2d_65111_9073705
conv2d_65111_9073707
dense_37989_9073747
dense_37989_9073749
dense_37990_9073774
dense_37990_9073776
dense_37991_9073831
dense_37991_9073833
identityЂ$conv2d_65109/StatefulPartitionedCallЂ$conv2d_65110/StatefulPartitionedCallЂ$conv2d_65111/StatefulPartitionedCallЂ#dense_37989/StatefulPartitionedCallЂ#dense_37990/StatefulPartitionedCallЂ#dense_37991/StatefulPartitionedCallЂ!dropout_7/StatefulPartitionedCallФ
$conv2d_65109/StatefulPartitionedCallStatefulPartitionedCallconv2d_65109_inputconv2d_65109_9073649conv2d_65109_9073651*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_90736382&
$conv2d_65109/StatefulPartitionedCallЊ
#max_pooling2d_32565/PartitionedCallPartitionedCall-conv2d_65109/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_90735932%
#max_pooling2d_32565/PartitionedCallо
$conv2d_65110/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32565/PartitionedCall:output:0conv2d_65110_9073677conv2d_65110_9073679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_90736662&
$conv2d_65110/StatefulPartitionedCallЈ
#max_pooling2d_32566/PartitionedCallPartitionedCall-conv2d_65110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_90736052%
#max_pooling2d_32566/PartitionedCallн
$conv2d_65111/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_32566/PartitionedCall:output:0conv2d_65111_9073705conv2d_65111_9073707*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_90736942&
$conv2d_65111/StatefulPartitionedCallЉ
#max_pooling2d_32567/PartitionedCallPartitionedCall-conv2d_65111/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_90736172%
#max_pooling2d_32567/PartitionedCall
flatten_10855/PartitionedCallPartitionedCall,max_pooling2d_32567/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџЙ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_flatten_10855_layer_call_and_return_conditional_losses_90737172
flatten_10855/PartitionedCallЩ
#dense_37989/StatefulPartitionedCallStatefulPartitionedCall&flatten_10855/PartitionedCall:output:0dense_37989_9073747dense_37989_9073749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37989_layer_call_and_return_conditional_losses_90737362%
#dense_37989/StatefulPartitionedCallЯ
#dense_37990/StatefulPartitionedCallStatefulPartitionedCall,dense_37989/StatefulPartitionedCall:output:0dense_37990_9073774dense_37990_9073776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37990_layer_call_and_return_conditional_losses_90737632%
#dense_37990/StatefulPartitionedCall
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall,dense_37990/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_90737912#
!dropout_7/StatefulPartitionedCallЭ
#dense_37991/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_37991_9073831dense_37991_9073833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37991_layer_call_and_return_conditional_losses_90738202%
#dense_37991/StatefulPartitionedCall
IdentityIdentity,dense_37991/StatefulPartitionedCall:output:0%^conv2d_65109/StatefulPartitionedCall%^conv2d_65110/StatefulPartitionedCall%^conv2d_65111/StatefulPartitionedCall$^dense_37989/StatefulPartitionedCall$^dense_37990/StatefulPartitionedCall$^dense_37991/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_65109/StatefulPartitionedCall$conv2d_65109/StatefulPartitionedCall2L
$conv2d_65110/StatefulPartitionedCall$conv2d_65110/StatefulPartitionedCall2L
$conv2d_65111/StatefulPartitionedCall$conv2d_65111/StatefulPartitionedCall2J
#dense_37989/StatefulPartitionedCall#dense_37989/StatefulPartitionedCall2J
#dense_37990/StatefulPartitionedCall#dense_37990/StatefulPartitionedCall2J
#dense_37991/StatefulPartitionedCall#dense_37991/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_65109_input

l
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_9073593

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
й

т
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_9074272

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@@
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_9073617

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
f
J__inference_flatten_10855_layer_call_and_return_conditional_losses_9073717

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_9074349

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ<:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs

e
F__inference_dropout_7_layer_call_and_return_conditional_losses_9074344

inputs
identityc
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
:џџџџџџџџџ<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ<:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_9073796

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ<:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_9073605

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
К
Q
5__inference_max_pooling2d_32566_layer_call_fn_9073611

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_90736052
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
МQ
№
I__inference_sequential_7_layer_call_and_return_conditional_losses_9074111

inputs/
+conv2d_65109_conv2d_readvariableop_resource0
,conv2d_65109_biasadd_readvariableop_resource/
+conv2d_65110_conv2d_readvariableop_resource0
,conv2d_65110_biasadd_readvariableop_resource/
+conv2d_65111_conv2d_readvariableop_resource0
,conv2d_65111_biasadd_readvariableop_resource.
*dense_37989_matmul_readvariableop_resource/
+dense_37989_biasadd_readvariableop_resource.
*dense_37990_matmul_readvariableop_resource/
+dense_37990_biasadd_readvariableop_resource.
*dense_37991_matmul_readvariableop_resource/
+dense_37991_biasadd_readvariableop_resource
identityЂ#conv2d_65109/BiasAdd/ReadVariableOpЂ"conv2d_65109/Conv2D/ReadVariableOpЂ#conv2d_65110/BiasAdd/ReadVariableOpЂ"conv2d_65110/Conv2D/ReadVariableOpЂ#conv2d_65111/BiasAdd/ReadVariableOpЂ"conv2d_65111/Conv2D/ReadVariableOpЂ"dense_37989/BiasAdd/ReadVariableOpЂ!dense_37989/MatMul/ReadVariableOpЂ"dense_37990/BiasAdd/ReadVariableOpЂ!dense_37990/MatMul/ReadVariableOpЂ"dense_37991/BiasAdd/ReadVariableOpЂ!dense_37991/MatMul/ReadVariableOpМ
"conv2d_65109/Conv2D/ReadVariableOpReadVariableOp+conv2d_65109_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"conv2d_65109/Conv2D/ReadVariableOpЬ
conv2d_65109/Conv2DConv2Dinputs*conv2d_65109/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d_65109/Conv2DГ
#conv2d_65109/BiasAdd/ReadVariableOpReadVariableOp,conv2d_65109_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv2d_65109/BiasAdd/ReadVariableOpО
conv2d_65109/BiasAddBiasAddconv2d_65109/Conv2D:output:0+conv2d_65109/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_65109/BiasAdd
conv2d_65109/ReluReluconv2d_65109/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_65109/Reluе
max_pooling2d_32565/MaxPoolMaxPoolconv2d_65109/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_32565/MaxPoolМ
"conv2d_65110/Conv2D/ReadVariableOpReadVariableOp+conv2d_65110_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"conv2d_65110/Conv2D/ReadVariableOpъ
conv2d_65110/Conv2DConv2D$max_pooling2d_32565/MaxPool:output:0*conv2d_65110/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_65110/Conv2DГ
#conv2d_65110/BiasAdd/ReadVariableOpReadVariableOp,conv2d_65110_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv2d_65110/BiasAdd/ReadVariableOpО
conv2d_65110/BiasAddBiasAddconv2d_65110/Conv2D:output:0+conv2d_65110/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_65110/BiasAdd
conv2d_65110/ReluReluconv2d_65110/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_65110/Reluг
max_pooling2d_32566/MaxPoolMaxPoolconv2d_65110/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_32566/MaxPoolН
"conv2d_65111/Conv2D/ReadVariableOpReadVariableOp+conv2d_65111_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"conv2d_65111/Conv2D/ReadVariableOpщ
conv2d_65111/Conv2DConv2D$max_pooling2d_32566/MaxPool:output:0*conv2d_65111/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_65111/Conv2DД
#conv2d_65111/BiasAdd/ReadVariableOpReadVariableOp,conv2d_65111_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#conv2d_65111/BiasAdd/ReadVariableOpН
conv2d_65111/BiasAddBiasAddconv2d_65111/Conv2D:output:0+conv2d_65111/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_65111/BiasAdd
conv2d_65111/ReluReluconv2d_65111/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_65111/Reluд
max_pooling2d_32567/MaxPoolMaxPoolconv2d_65111/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_32567/MaxPool{
flatten_10855/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2
flatten_10855/ConstБ
flatten_10855/ReshapeReshape$max_pooling2d_32567/MaxPool:output:0flatten_10855/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2
flatten_10855/ReshapeГ
!dense_37989/MatMul/ReadVariableOpReadVariableOp*dense_37989_matmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype02#
!dense_37989/MatMul/ReadVariableOpЏ
dense_37989/MatMulMatMulflatten_10855/Reshape:output:0)dense_37989/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_37989/MatMulА
"dense_37989/BiasAdd/ReadVariableOpReadVariableOp+dense_37989_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02$
"dense_37989/BiasAdd/ReadVariableOpБ
dense_37989/BiasAddBiasAdddense_37989/MatMul:product:0*dense_37989/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_37989/BiasAdd|
dense_37989/ReluReludense_37989/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_37989/ReluБ
!dense_37990/MatMul/ReadVariableOpReadVariableOp*dense_37990_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02#
!dense_37990/MatMul/ReadVariableOpЏ
dense_37990/MatMulMatMuldense_37989/Relu:activations:0)dense_37990/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_37990/MatMulА
"dense_37990/BiasAdd/ReadVariableOpReadVariableOp+dense_37990_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dense_37990/BiasAdd/ReadVariableOpБ
dense_37990/BiasAddBiasAdddense_37990/MatMul:product:0*dense_37990/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_37990/BiasAdd|
dense_37990/ReluReludense_37990/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_37990/Reluw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/ConstЉ
dropout_7/dropout/MulMuldense_37990/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout_7/dropout/Mul
dropout_7/dropout/ShapeShapedense_37990/Relu:activations:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shapeв
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/yц
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
dropout_7/dropout/GreaterEqual
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ<2
dropout_7/dropout/CastЂ
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout_7/dropout/Mul_1Б
!dense_37991/MatMul/ReadVariableOpReadVariableOp*dense_37991_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!dense_37991/MatMul/ReadVariableOpЌ
dense_37991/MatMulMatMuldropout_7/dropout/Mul_1:z:0)dense_37991/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_37991/MatMulА
"dense_37991/BiasAdd/ReadVariableOpReadVariableOp+dense_37991_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_37991/BiasAdd/ReadVariableOpБ
dense_37991/BiasAddBiasAdddense_37991/MatMul:product:0*dense_37991/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_37991/BiasAdd
dense_37991/SoftmaxSoftmaxdense_37991/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_37991/Softmax­
IdentityIdentitydense_37991/Softmax:softmax:0$^conv2d_65109/BiasAdd/ReadVariableOp#^conv2d_65109/Conv2D/ReadVariableOp$^conv2d_65110/BiasAdd/ReadVariableOp#^conv2d_65110/Conv2D/ReadVariableOp$^conv2d_65111/BiasAdd/ReadVariableOp#^conv2d_65111/Conv2D/ReadVariableOp#^dense_37989/BiasAdd/ReadVariableOp"^dense_37989/MatMul/ReadVariableOp#^dense_37990/BiasAdd/ReadVariableOp"^dense_37990/MatMul/ReadVariableOp#^dense_37991/BiasAdd/ReadVariableOp"^dense_37991/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2J
#conv2d_65109/BiasAdd/ReadVariableOp#conv2d_65109/BiasAdd/ReadVariableOp2H
"conv2d_65109/Conv2D/ReadVariableOp"conv2d_65109/Conv2D/ReadVariableOp2J
#conv2d_65110/BiasAdd/ReadVariableOp#conv2d_65110/BiasAdd/ReadVariableOp2H
"conv2d_65110/Conv2D/ReadVariableOp"conv2d_65110/Conv2D/ReadVariableOp2J
#conv2d_65111/BiasAdd/ReadVariableOp#conv2d_65111/BiasAdd/ReadVariableOp2H
"conv2d_65111/Conv2D/ReadVariableOp"conv2d_65111/Conv2D/ReadVariableOp2H
"dense_37989/BiasAdd/ReadVariableOp"dense_37989/BiasAdd/ReadVariableOp2F
!dense_37989/MatMul/ReadVariableOp!dense_37989/MatMul/ReadVariableOp2H
"dense_37990/BiasAdd/ReadVariableOp"dense_37990/BiasAdd/ReadVariableOp2F
!dense_37990/MatMul/ReadVariableOp!dense_37990/MatMul/ReadVariableOp2H
"dense_37991/BiasAdd/ReadVariableOp"dense_37991/BiasAdd/ReadVariableOp2F
!dense_37991/MatMul/ReadVariableOp!dense_37991/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЇП

#__inference__traced_restore_9074682
file_prefix(
$assignvariableop_conv2d_65109_kernel(
$assignvariableop_1_conv2d_65109_bias*
&assignvariableop_2_conv2d_65110_kernel(
$assignvariableop_3_conv2d_65110_bias*
&assignvariableop_4_conv2d_65111_kernel(
$assignvariableop_5_conv2d_65111_bias)
%assignvariableop_6_dense_37989_kernel'
#assignvariableop_7_dense_37989_bias)
%assignvariableop_8_dense_37990_kernel'
#assignvariableop_9_dense_37990_bias*
&assignvariableop_10_dense_37991_kernel(
$assignvariableop_11_dense_37991_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_12
.assignvariableop_21_adam_conv2d_65109_kernel_m0
,assignvariableop_22_adam_conv2d_65109_bias_m2
.assignvariableop_23_adam_conv2d_65110_kernel_m0
,assignvariableop_24_adam_conv2d_65110_bias_m2
.assignvariableop_25_adam_conv2d_65111_kernel_m0
,assignvariableop_26_adam_conv2d_65111_bias_m1
-assignvariableop_27_adam_dense_37989_kernel_m/
+assignvariableop_28_adam_dense_37989_bias_m1
-assignvariableop_29_adam_dense_37990_kernel_m/
+assignvariableop_30_adam_dense_37990_bias_m1
-assignvariableop_31_adam_dense_37991_kernel_m/
+assignvariableop_32_adam_dense_37991_bias_m2
.assignvariableop_33_adam_conv2d_65109_kernel_v0
,assignvariableop_34_adam_conv2d_65109_bias_v2
.assignvariableop_35_adam_conv2d_65110_kernel_v0
,assignvariableop_36_adam_conv2d_65110_bias_v2
.assignvariableop_37_adam_conv2d_65111_kernel_v0
,assignvariableop_38_adam_conv2d_65111_bias_v1
-assignvariableop_39_adam_dense_37989_kernel_v/
+assignvariableop_40_adam_dense_37989_bias_v1
-assignvariableop_41_adam_dense_37990_kernel_v/
+assignvariableop_42_adam_dense_37990_bias_v1
-assignvariableop_43_adam_dense_37991_kernel_v/
+assignvariableop_44_adam_dense_37991_bias_v
identity_46ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЃ
AssignVariableOpAssignVariableOp$assignvariableop_conv2d_65109_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Љ
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_65109_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ћ
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv2d_65110_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Љ
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv2d_65110_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ћ
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv2d_65111_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Љ
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv2d_65111_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Њ
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense_37989_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ј
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_37989_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Њ
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_37990_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ј
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_37990_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_dense_37991_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ќ
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_37991_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12Ѕ
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ї
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ї
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15І
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ў
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ё
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ё
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѓ
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ж
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_conv2d_65109_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Д
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_conv2d_65109_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ж
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_conv2d_65110_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Д
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_conv2d_65110_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ж
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_conv2d_65111_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Д
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_conv2d_65111_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Е
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_dense_37989_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Г
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_dense_37989_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Е
AssignVariableOp_29AssignVariableOp-assignvariableop_29_adam_dense_37990_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Г
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_37990_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Е
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_37991_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Г
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_37991_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ж
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_conv2d_65109_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Д
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv2d_65109_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ж
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_conv2d_65110_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Д
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv2d_65110_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ж
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_conv2d_65111_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Д
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_conv2d_65111_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Е
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_dense_37989_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Г
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_dense_37989_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Е
AssignVariableOp_41AssignVariableOp-assignvariableop_41_adam_dense_37990_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Г
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_dense_37990_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Е
AssignVariableOp_43AssignVariableOp-assignvariableop_43_adam_dense_37991_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Г
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_dense_37991_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpМ
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45Џ
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*Ы
_input_shapesЙ
Ж: :::::::::::::::::::::::::::::::::::::::::::::2$
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
п

т
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_9074252

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
п

т
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_9073638

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш

-__inference_dense_37991_layer_call_fn_9074379

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_37991_layer_call_and_return_conditional_losses_90738202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ю
serving_defaultК
[
conv2d_65109_inputE
$serving_default_conv2d_65109_input:0џџџџџџџџџ?
dense_379910
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ло
лV
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
Ў__call__
+Џ&call_and_return_all_conditional_losses
А_default_save_signature"сR
_tf_keras_sequentialТR{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_65109_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_65109", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32565", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_65110", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32566", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_65111", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32567", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_10855", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_37989", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37990", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_37991", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_65109_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_65109", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32565", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_65110", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32566", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_65111", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32567", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_10855", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_37989", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37990", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_37991", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"к	
_tf_keras_layerР	{"class_name": "Conv2D", "name": "conv2d_65109", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_65109", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}

regularization_losses
	variables
trainable_variables
	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d_32565", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_32565", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ў	

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Conv2D", "name": "conv2d_65110", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_65110", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}

"regularization_losses
#	variables
$trainable_variables
%	keras_api
З__call__
+И&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d_32566", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_32566", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
§	

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Conv2D", "name": "conv2d_65111", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_65111", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

,regularization_losses
-	variables
.trainable_variables
/	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d_32567", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_32567", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
№
0regularization_losses
1	variables
2trainable_variables
3	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"class_name": "Flatten", "name": "flatten_10855", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_10855", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"к
_tf_keras_layerР{"class_name": "Dense", "name": "dense_37989", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37989", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 56448}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56448]}}
ќ

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"class_name": "Dense", "name": "dense_37990", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37990", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
ч
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ќ

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"class_name": "Dense", "name": "dense_37991", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37991", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
У
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemmmm&m'm4m5m:m;mDm EmЁvЂvЃvЄvЅ&vІ'vЇ4vЈ5vЉ:vЊ;vЋDvЌEv­"
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
Ю
Ometrics

Players
regularization_losses
Qlayer_regularization_losses
	variables
trainable_variables
Rnon_trainable_variables
Slayer_metrics
Ў__call__
А_default_save_signature
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
-
Чserving_default"
signature_map
-:+ 2conv2d_65109/kernel
: 2conv2d_65109/bias
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
А
Tmetrics

Ulayers
regularization_losses
Vlayer_regularization_losses
	variables
trainable_variables
Wnon_trainable_variables
Xlayer_metrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Ymetrics

Zlayers
regularization_losses
[layer_regularization_losses
	variables
trainable_variables
\non_trainable_variables
]layer_metrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
-:+ @2conv2d_65110/kernel
:@2conv2d_65110/bias
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
А
^metrics

_layers
regularization_losses
`layer_regularization_losses
	variables
 trainable_variables
anon_trainable_variables
blayer_metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
cmetrics

dlayers
"regularization_losses
elayer_regularization_losses
#	variables
$trainable_variables
fnon_trainable_variables
glayer_metrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
.:,@2conv2d_65111/kernel
 :2conv2d_65111/bias
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
А
hmetrics

ilayers
(regularization_losses
jlayer_regularization_losses
)	variables
*trainable_variables
knon_trainable_variables
llayer_metrics
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
mmetrics

nlayers
,regularization_losses
olayer_regularization_losses
-	variables
.trainable_variables
pnon_trainable_variables
qlayer_metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
rmetrics

slayers
0regularization_losses
tlayer_regularization_losses
1	variables
2trainable_variables
unon_trainable_variables
vlayer_metrics
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
&:$
Йx2dense_37989/kernel
:x2dense_37989/bias
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
А
wmetrics

xlayers
6regularization_losses
ylayer_regularization_losses
7	variables
8trainable_variables
znon_trainable_variables
{layer_metrics
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
$:"x<2dense_37990/kernel
:<2dense_37990/bias
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
Б
|metrics

}layers
<regularization_losses
~layer_regularization_losses
=	variables
>trainable_variables
non_trainable_variables
layer_metrics
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
layers
@regularization_losses
 layer_regularization_losses
A	variables
Btrainable_variables
non_trainable_variables
layer_metrics
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
$:"<2dense_37991/kernel
:2dense_37991/bias
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
Е
metrics
layers
Fregularization_losses
 layer_regularization_losses
G	variables
Htrainable_variables
non_trainable_variables
layer_metrics
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
0
1"
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
П

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


total

count

_fn_kwargs
	variables
	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
2:0 2Adam/conv2d_65109/kernel/m
$:" 2Adam/conv2d_65109/bias/m
2:0 @2Adam/conv2d_65110/kernel/m
$:"@2Adam/conv2d_65110/bias/m
3:1@2Adam/conv2d_65111/kernel/m
%:#2Adam/conv2d_65111/bias/m
+:)
Йx2Adam/dense_37989/kernel/m
#:!x2Adam/dense_37989/bias/m
):'x<2Adam/dense_37990/kernel/m
#:!<2Adam/dense_37990/bias/m
):'<2Adam/dense_37991/kernel/m
#:!2Adam/dense_37991/bias/m
2:0 2Adam/conv2d_65109/kernel/v
$:" 2Adam/conv2d_65109/bias/v
2:0 @2Adam/conv2d_65110/kernel/v
$:"@2Adam/conv2d_65110/bias/v
3:1@2Adam/conv2d_65111/kernel/v
%:#2Adam/conv2d_65111/bias/v
+:)
Йx2Adam/dense_37989/kernel/v
#:!x2Adam/dense_37989/bias/v
):'x<2Adam/dense_37990/kernel/v
#:!<2Adam/dense_37990/bias/v
):'<2Adam/dense_37991/kernel/v
#:!2Adam/dense_37991/bias/v
2
.__inference_sequential_7_layer_call_fn_9074192
.__inference_sequential_7_layer_call_fn_9073945
.__inference_sequential_7_layer_call_fn_9074013
.__inference_sequential_7_layer_call_fn_9074221Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
I__inference_sequential_7_layer_call_and_return_conditional_losses_9073876
I__inference_sequential_7_layer_call_and_return_conditional_losses_9074111
I__inference_sequential_7_layer_call_and_return_conditional_losses_9073837
I__inference_sequential_7_layer_call_and_return_conditional_losses_9074163Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ѕ2ђ
"__inference__wrapped_model_9073587Ы
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *;Ђ8
63
conv2d_65109_inputџџџџџџџџџ
и2е
.__inference_conv2d_65109_layer_call_fn_9074241Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_9074232Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
5__inference_max_pooling2d_32565_layer_call_fn_9073599р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
И2Е
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_9073593р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
и2е
.__inference_conv2d_65110_layer_call_fn_9074261Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_9074252Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
5__inference_max_pooling2d_32566_layer_call_fn_9073611р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
И2Е
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_9073605р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
и2е
.__inference_conv2d_65111_layer_call_fn_9074281Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_9074272Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
5__inference_max_pooling2d_32567_layer_call_fn_9073623р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
И2Е
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_9073617р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
й2ж
/__inference_flatten_10855_layer_call_fn_9074292Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_flatten_10855_layer_call_and_return_conditional_losses_9074287Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_dense_37989_layer_call_fn_9074312Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_dense_37989_layer_call_and_return_conditional_losses_9074303Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_dense_37990_layer_call_fn_9074332Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_dense_37990_layer_call_and_return_conditional_losses_9074323Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
+__inference_dropout_7_layer_call_fn_9074359
+__inference_dropout_7_layer_call_fn_9074354Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ъ2Ч
F__inference_dropout_7_layer_call_and_return_conditional_losses_9074344
F__inference_dropout_7_layer_call_and_return_conditional_losses_9074349Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
з2д
-__inference_dense_37991_layer_call_fn_9074379Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_dense_37991_layer_call_and_return_conditional_losses_9074370Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
зBд
%__inference_signature_wrapper_9074052conv2d_65109_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 З
"__inference__wrapped_model_9073587&'45:;DEEЂB
;Ђ8
63
conv2d_65109_inputџџџџџџџџџ
Њ "9Њ6
4
dense_37991%"
dense_37991џџџџџџџџџН
I__inference_conv2d_65109_layer_call_and_return_conditional_losses_9074232p9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ 
 
.__inference_conv2d_65109_layer_call_fn_9074241c9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџ Н
I__inference_conv2d_65110_layer_call_and_return_conditional_losses_9074252p9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ 
Њ "/Ђ,
%"
0џџџџџџџџџ@
 
.__inference_conv2d_65110_layer_call_fn_9074261c9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ 
Њ ""џџџџџџџџџ@К
I__inference_conv2d_65111_layer_call_and_return_conditional_losses_9074272m&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@@
Њ ".Ђ+
$!
0џџџџџџџџџ@@
 
.__inference_conv2d_65111_layer_call_fn_9074281`&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@@
Њ "!џџџџџџџџџ@@Њ
H__inference_dense_37989_layer_call_and_return_conditional_losses_9074303^451Ђ.
'Ђ$
"
inputsџџџџџџџџџЙ
Њ "%Ђ"

0џџџџџџџџџx
 
-__inference_dense_37989_layer_call_fn_9074312Q451Ђ.
'Ђ$
"
inputsџџџџџџџџџЙ
Њ "џџџџџџџџџxЈ
H__inference_dense_37990_layer_call_and_return_conditional_losses_9074323\:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "%Ђ"

0џџџџџџџџџ<
 
-__inference_dense_37990_layer_call_fn_9074332O:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "џџџџџџџџџ<Ј
H__inference_dense_37991_layer_call_and_return_conditional_losses_9074370\DE/Ђ,
%Ђ"
 
inputsџџџџџџџџџ<
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_dense_37991_layer_call_fn_9074379ODE/Ђ,
%Ђ"
 
inputsџџџџџџџџџ<
Њ "џџџџџџџџџІ
F__inference_dropout_7_layer_call_and_return_conditional_losses_9074344\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p
Њ "%Ђ"

0џџџџџџџџџ<
 І
F__inference_dropout_7_layer_call_and_return_conditional_losses_9074349\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p 
Њ "%Ђ"

0џџџџџџџџџ<
 ~
+__inference_dropout_7_layer_call_fn_9074354O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p
Њ "џџџџџџџџџ<~
+__inference_dropout_7_layer_call_fn_9074359O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p 
Њ "џџџџџџџџџ<Б
J__inference_flatten_10855_layer_call_and_return_conditional_losses_9074287c8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "'Ђ$

0џџџџџџџџџЙ
 
/__inference_flatten_10855_layer_call_fn_9074292V8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "џџџџџџџџџЙѓ
P__inference_max_pooling2d_32565_layer_call_and_return_conditional_losses_9073593RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
5__inference_max_pooling2d_32565_layer_call_fn_9073599RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
P__inference_max_pooling2d_32566_layer_call_and_return_conditional_losses_9073605RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
5__inference_max_pooling2d_32566_layer_call_fn_9073611RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
P__inference_max_pooling2d_32567_layer_call_and_return_conditional_losses_9073617RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
5__inference_max_pooling2d_32567_layer_call_fn_9073623RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџв
I__inference_sequential_7_layer_call_and_return_conditional_losses_9073837&'45:;DEMЂJ
CЂ@
63
conv2d_65109_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 в
I__inference_sequential_7_layer_call_and_return_conditional_losses_9073876&'45:;DEMЂJ
CЂ@
63
conv2d_65109_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
I__inference_sequential_7_layer_call_and_return_conditional_losses_9074111x&'45:;DEAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
I__inference_sequential_7_layer_call_and_return_conditional_losses_9074163x&'45:;DEAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Љ
.__inference_sequential_7_layer_call_fn_9073945w&'45:;DEMЂJ
CЂ@
63
conv2d_65109_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџЉ
.__inference_sequential_7_layer_call_fn_9074013w&'45:;DEMЂJ
CЂ@
63
conv2d_65109_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_7_layer_call_fn_9074192k&'45:;DEAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_7_layer_call_fn_9074221k&'45:;DEAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџа
%__inference_signature_wrapper_9074052І&'45:;DE[ЂX
Ђ 
QЊN
L
conv2d_65109_input63
conv2d_65109_inputџџџџџџџџџ"9Њ6
4
dense_37991%"
dense_37991џџџџџџџџџ