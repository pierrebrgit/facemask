са
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8м	

conv2d_33801/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv2d_33801/kernel

'conv2d_33801/kernel/Read/ReadVariableOpReadVariableOpconv2d_33801/kernel*&
_output_shapes
: *
dtype0
z
conv2d_33801/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_33801/bias
s
%conv2d_33801/bias/Read/ReadVariableOpReadVariableOpconv2d_33801/bias*
_output_shapes
: *
dtype0

conv2d_33802/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameconv2d_33802/kernel

'conv2d_33802/kernel/Read/ReadVariableOpReadVariableOpconv2d_33802/kernel*&
_output_shapes
: @*
dtype0
z
conv2d_33802/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_33802/bias
s
%conv2d_33802/bias/Read/ReadVariableOpReadVariableOpconv2d_33802/bias*
_output_shapes
:@*
dtype0

conv2d_33803/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameconv2d_33803/kernel

'conv2d_33803/kernel/Read/ReadVariableOpReadVariableOpconv2d_33803/kernel*'
_output_shapes
:@*
dtype0
{
conv2d_33803/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_33803/bias
t
%conv2d_33803/bias/Read/ReadVariableOpReadVariableOpconv2d_33803/bias*
_output_shapes	
:*
dtype0

dense_19721/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Йx*#
shared_namedense_19721/kernel
{
&dense_19721/kernel/Read/ReadVariableOpReadVariableOpdense_19721/kernel* 
_output_shapes
:
Йx*
dtype0
x
dense_19721/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*!
shared_namedense_19721/bias
q
$dense_19721/bias/Read/ReadVariableOpReadVariableOpdense_19721/bias*
_output_shapes
:x*
dtype0

dense_19722/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*#
shared_namedense_19722/kernel
y
&dense_19722/kernel/Read/ReadVariableOpReadVariableOpdense_19722/kernel*
_output_shapes

:x<*
dtype0
x
dense_19722/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_namedense_19722/bias
q
$dense_19722/bias/Read/ReadVariableOpReadVariableOpdense_19722/bias*
_output_shapes
:<*
dtype0

dense_19723/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*#
shared_namedense_19723/kernel
y
&dense_19723/kernel/Read/ReadVariableOpReadVariableOpdense_19723/kernel*
_output_shapes

:<*
dtype0
x
dense_19723/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_19723/bias
q
$dense_19723/bias/Read/ReadVariableOpReadVariableOpdense_19723/bias*
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
Adam/conv2d_33801/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv2d_33801/kernel/m

.Adam/conv2d_33801/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33801/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_33801/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_33801/bias/m

,Adam/conv2d_33801/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33801/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_33802/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv2d_33802/kernel/m

.Adam/conv2d_33802/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33802/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_33802/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_33802/bias/m

,Adam/conv2d_33802/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33802/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_33803/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/conv2d_33803/kernel/m

.Adam/conv2d_33803/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33803/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_33803/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_33803/bias/m

,Adam/conv2d_33803/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33803/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_19721/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Йx**
shared_nameAdam/dense_19721/kernel/m

-Adam/dense_19721/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19721/kernel/m* 
_output_shapes
:
Йx*
dtype0

Adam/dense_19721/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*(
shared_nameAdam/dense_19721/bias/m

+Adam/dense_19721/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19721/bias/m*
_output_shapes
:x*
dtype0

Adam/dense_19722/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<**
shared_nameAdam/dense_19722/kernel/m

-Adam/dense_19722/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19722/kernel/m*
_output_shapes

:x<*
dtype0

Adam/dense_19722/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dense_19722/bias/m

+Adam/dense_19722/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19722/bias/m*
_output_shapes
:<*
dtype0

Adam/dense_19723/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<**
shared_nameAdam/dense_19723/kernel/m

-Adam/dense_19723/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19723/kernel/m*
_output_shapes

:<*
dtype0

Adam/dense_19723/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_19723/bias/m

+Adam/dense_19723/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19723/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_33801/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv2d_33801/kernel/v

.Adam/conv2d_33801/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33801/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_33801/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_33801/bias/v

,Adam/conv2d_33801/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33801/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_33802/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv2d_33802/kernel/v

.Adam/conv2d_33802/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33802/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_33802/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_33802/bias/v

,Adam/conv2d_33802/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33802/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_33803/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/conv2d_33803/kernel/v

.Adam/conv2d_33803/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33803/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_33803/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_33803/bias/v

,Adam/conv2d_33803/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33803/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_19721/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Йx**
shared_nameAdam/dense_19721/kernel/v

-Adam/dense_19721/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19721/kernel/v* 
_output_shapes
:
Йx*
dtype0

Adam/dense_19721/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*(
shared_nameAdam/dense_19721/bias/v

+Adam/dense_19721/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19721/bias/v*
_output_shapes
:x*
dtype0

Adam/dense_19722/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<**
shared_nameAdam/dense_19722/kernel/v

-Adam/dense_19722/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19722/kernel/v*
_output_shapes

:x<*
dtype0

Adam/dense_19722/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameAdam/dense_19722/bias/v

+Adam/dense_19722/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19722/bias/v*
_output_shapes
:<*
dtype0

Adam/dense_19723/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<**
shared_nameAdam/dense_19723/kernel/v

-Adam/dense_19723/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19723/kernel/v*
_output_shapes

:<*
dtype0

Adam/dense_19723/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_19723/bias/v

+Adam/dense_19723/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19723/bias/v*
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
VARIABLE_VALUEconv2d_33801/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_33801/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_33802/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_33802/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_33803/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_33803/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_19721/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_19721/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_19722/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_19722/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_19723/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_19723/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/conv2d_33801/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_33801/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_33802/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_33802/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_33803/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_33803/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_19721/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_19721/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_19722/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_19722/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_19723/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_19723/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_33801/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_33801/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_33802/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_33802/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_33803/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_33803/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_19721/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_19721/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_19722/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_19722/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_19723/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_19723/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

"serving_default_conv2d_33801_inputPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
У
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conv2d_33801_inputconv2d_33801/kernelconv2d_33801/biasconv2d_33802/kernelconv2d_33802/biasconv2d_33803/kernelconv2d_33803/biasdense_19721/kerneldense_19721/biasdense_19722/kerneldense_19722/biasdense_19723/kerneldense_19723/bias*
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
%__inference_signature_wrapper_4795467
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'conv2d_33801/kernel/Read/ReadVariableOp%conv2d_33801/bias/Read/ReadVariableOp'conv2d_33802/kernel/Read/ReadVariableOp%conv2d_33802/bias/Read/ReadVariableOp'conv2d_33803/kernel/Read/ReadVariableOp%conv2d_33803/bias/Read/ReadVariableOp&dense_19721/kernel/Read/ReadVariableOp$dense_19721/bias/Read/ReadVariableOp&dense_19722/kernel/Read/ReadVariableOp$dense_19722/bias/Read/ReadVariableOp&dense_19723/kernel/Read/ReadVariableOp$dense_19723/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/conv2d_33801/kernel/m/Read/ReadVariableOp,Adam/conv2d_33801/bias/m/Read/ReadVariableOp.Adam/conv2d_33802/kernel/m/Read/ReadVariableOp,Adam/conv2d_33802/bias/m/Read/ReadVariableOp.Adam/conv2d_33803/kernel/m/Read/ReadVariableOp,Adam/conv2d_33803/bias/m/Read/ReadVariableOp-Adam/dense_19721/kernel/m/Read/ReadVariableOp+Adam/dense_19721/bias/m/Read/ReadVariableOp-Adam/dense_19722/kernel/m/Read/ReadVariableOp+Adam/dense_19722/bias/m/Read/ReadVariableOp-Adam/dense_19723/kernel/m/Read/ReadVariableOp+Adam/dense_19723/bias/m/Read/ReadVariableOp.Adam/conv2d_33801/kernel/v/Read/ReadVariableOp,Adam/conv2d_33801/bias/v/Read/ReadVariableOp.Adam/conv2d_33802/kernel/v/Read/ReadVariableOp,Adam/conv2d_33802/bias/v/Read/ReadVariableOp.Adam/conv2d_33803/kernel/v/Read/ReadVariableOp,Adam/conv2d_33803/bias/v/Read/ReadVariableOp-Adam/dense_19721/kernel/v/Read/ReadVariableOp+Adam/dense_19721/bias/v/Read/ReadVariableOp-Adam/dense_19722/kernel/v/Read/ReadVariableOp+Adam/dense_19722/bias/v/Read/ReadVariableOp-Adam/dense_19723/kernel/v/Read/ReadVariableOp+Adam/dense_19723/bias/v/Read/ReadVariableOpConst*:
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
 __inference__traced_save_4795952


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_33801/kernelconv2d_33801/biasconv2d_33802/kernelconv2d_33802/biasconv2d_33803/kernelconv2d_33803/biasdense_19721/kerneldense_19721/biasdense_19722/kerneldense_19722/biasdense_19723/kerneldense_19723/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_33801/kernel/mAdam/conv2d_33801/bias/mAdam/conv2d_33802/kernel/mAdam/conv2d_33802/bias/mAdam/conv2d_33803/kernel/mAdam/conv2d_33803/bias/mAdam/dense_19721/kernel/mAdam/dense_19721/bias/mAdam/dense_19722/kernel/mAdam/dense_19722/bias/mAdam/dense_19723/kernel/mAdam/dense_19723/bias/mAdam/conv2d_33801/kernel/vAdam/conv2d_33801/bias/vAdam/conv2d_33802/kernel/vAdam/conv2d_33802/bias/vAdam/conv2d_33803/kernel/vAdam/conv2d_33803/bias/vAdam/dense_19721/kernel/vAdam/dense_19721/bias/vAdam/dense_19722/kernel/vAdam/dense_19722/bias/vAdam/dense_19723/kernel/vAdam/dense_19723/bias/v*9
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
#__inference__traced_restore_4796097ѕ
3

I__inference_sequential_3_layer_call_and_return_conditional_losses_4795401

inputs
conv2d_33801_4795365
conv2d_33801_4795367
conv2d_33802_4795371
conv2d_33802_4795373
conv2d_33803_4795377
conv2d_33803_4795379
dense_19721_4795384
dense_19721_4795386
dense_19722_4795389
dense_19722_4795391
dense_19723_4795395
dense_19723_4795397
identityЂ$conv2d_33801/StatefulPartitionedCallЂ$conv2d_33802/StatefulPartitionedCallЂ$conv2d_33803/StatefulPartitionedCallЂ#dense_19721/StatefulPartitionedCallЂ#dense_19722/StatefulPartitionedCallЂ#dense_19723/StatefulPartitionedCallИ
$conv2d_33801/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_33801_4795365conv2d_33801_4795367*
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
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_47950532&
$conv2d_33801/StatefulPartitionedCallЊ
#max_pooling2d_16905/PartitionedCallPartitionedCall-conv2d_33801/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_47950082%
#max_pooling2d_16905/PartitionedCallо
$conv2d_33802/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16905/PartitionedCall:output:0conv2d_33802_4795371conv2d_33802_4795373*
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
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_47950812&
$conv2d_33802/StatefulPartitionedCallЈ
#max_pooling2d_16906/PartitionedCallPartitionedCall-conv2d_33802/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_47950202%
#max_pooling2d_16906/PartitionedCallн
$conv2d_33803/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16906/PartitionedCall:output:0conv2d_33803_4795377conv2d_33803_4795379*
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
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_47951092&
$conv2d_33803/StatefulPartitionedCallЉ
#max_pooling2d_16907/PartitionedCallPartitionedCall-conv2d_33803/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_47950322%
#max_pooling2d_16907/PartitionedCall
flatten_5635/PartitionedCallPartitionedCall,max_pooling2d_16907/PartitionedCall:output:0*
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
GPU2*0J 8 *R
fMRK
I__inference_flatten_5635_layer_call_and_return_conditional_losses_47951322
flatten_5635/PartitionedCallШ
#dense_19721/StatefulPartitionedCallStatefulPartitionedCall%flatten_5635/PartitionedCall:output:0dense_19721_4795384dense_19721_4795386*
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
H__inference_dense_19721_layer_call_and_return_conditional_losses_47951512%
#dense_19721/StatefulPartitionedCallЯ
#dense_19722/StatefulPartitionedCallStatefulPartitionedCall,dense_19721/StatefulPartitionedCall:output:0dense_19722_4795389dense_19722_4795391*
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
H__inference_dense_19722_layer_call_and_return_conditional_losses_47951782%
#dense_19722/StatefulPartitionedCall
dropout_3/PartitionedCallPartitionedCall,dense_19722/StatefulPartitionedCall:output:0*
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
F__inference_dropout_3_layer_call_and_return_conditional_losses_47952112
dropout_3/PartitionedCallХ
#dense_19723/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_19723_4795395dense_19723_4795397*
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
H__inference_dense_19723_layer_call_and_return_conditional_losses_47952352%
#dense_19723/StatefulPartitionedCallч
IdentityIdentity,dense_19723/StatefulPartitionedCall:output:0%^conv2d_33801/StatefulPartitionedCall%^conv2d_33802/StatefulPartitionedCall%^conv2d_33803/StatefulPartitionedCall$^dense_19721/StatefulPartitionedCall$^dense_19722/StatefulPartitionedCall$^dense_19723/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_33801/StatefulPartitionedCall$conv2d_33801/StatefulPartitionedCall2L
$conv2d_33802/StatefulPartitionedCall$conv2d_33802/StatefulPartitionedCall2L
$conv2d_33803/StatefulPartitionedCall$conv2d_33803/StatefulPartitionedCall2J
#dense_19721/StatefulPartitionedCall#dense_19721/StatefulPartitionedCall2J
#dense_19722/StatefulPartitionedCall#dense_19722/StatefulPartitionedCall2J
#dense_19723/StatefulPartitionedCall#dense_19723/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
e
I__inference_flatten_5635_layer_call_and_return_conditional_losses_4795132

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
ђ	
с
H__inference_dense_19722_layer_call_and_return_conditional_losses_4795738

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


.__inference_conv2d_33801_layer_call_fn_4795656

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
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_47950532
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
ш

-__inference_dense_19722_layer_call_fn_4795747

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
H__inference_dense_19722_layer_call_and_return_conditional_losses_47951782
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
ф	
І
.__inference_sequential_3_layer_call_fn_4795360
conv2d_33801_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_33801_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_3_layer_call_and_return_conditional_losses_47953332
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
_user_specified_nameconv2d_33801_input
ЇП

#__inference__traced_restore_4796097
file_prefix(
$assignvariableop_conv2d_33801_kernel(
$assignvariableop_1_conv2d_33801_bias*
&assignvariableop_2_conv2d_33802_kernel(
$assignvariableop_3_conv2d_33802_bias*
&assignvariableop_4_conv2d_33803_kernel(
$assignvariableop_5_conv2d_33803_bias)
%assignvariableop_6_dense_19721_kernel'
#assignvariableop_7_dense_19721_bias)
%assignvariableop_8_dense_19722_kernel'
#assignvariableop_9_dense_19722_bias*
&assignvariableop_10_dense_19723_kernel(
$assignvariableop_11_dense_19723_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_12
.assignvariableop_21_adam_conv2d_33801_kernel_m0
,assignvariableop_22_adam_conv2d_33801_bias_m2
.assignvariableop_23_adam_conv2d_33802_kernel_m0
,assignvariableop_24_adam_conv2d_33802_bias_m2
.assignvariableop_25_adam_conv2d_33803_kernel_m0
,assignvariableop_26_adam_conv2d_33803_bias_m1
-assignvariableop_27_adam_dense_19721_kernel_m/
+assignvariableop_28_adam_dense_19721_bias_m1
-assignvariableop_29_adam_dense_19722_kernel_m/
+assignvariableop_30_adam_dense_19722_bias_m1
-assignvariableop_31_adam_dense_19723_kernel_m/
+assignvariableop_32_adam_dense_19723_bias_m2
.assignvariableop_33_adam_conv2d_33801_kernel_v0
,assignvariableop_34_adam_conv2d_33801_bias_v2
.assignvariableop_35_adam_conv2d_33802_kernel_v0
,assignvariableop_36_adam_conv2d_33802_bias_v2
.assignvariableop_37_adam_conv2d_33803_kernel_v0
,assignvariableop_38_adam_conv2d_33803_bias_v1
-assignvariableop_39_adam_dense_19721_kernel_v/
+assignvariableop_40_adam_dense_19721_bias_v1
-assignvariableop_41_adam_dense_19722_kernel_v/
+assignvariableop_42_adam_dense_19722_bias_v1
-assignvariableop_43_adam_dense_19723_kernel_v/
+assignvariableop_44_adam_dense_19723_bias_v
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
AssignVariableOpAssignVariableOp$assignvariableop_conv2d_33801_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Љ
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_33801_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ћ
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv2d_33802_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Љ
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv2d_33802_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ћ
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv2d_33803_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Љ
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv2d_33803_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Њ
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense_19721_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ј
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_19721_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Њ
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_19722_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ј
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_19722_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_dense_19723_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ќ
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_19723_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_conv2d_33801_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Д
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_conv2d_33801_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ж
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_conv2d_33802_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Д
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_conv2d_33802_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ж
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_conv2d_33803_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Д
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_conv2d_33803_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Е
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_dense_19721_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Г
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_dense_19721_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Е
AssignVariableOp_29AssignVariableOp-assignvariableop_29_adam_dense_19722_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Г
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_19722_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Е
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_19723_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Г
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_19723_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ж
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_conv2d_33801_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Д
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv2d_33801_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ж
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_conv2d_33802_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Д
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv2d_33802_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ж
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_conv2d_33803_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Д
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_conv2d_33803_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Е
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_dense_19721_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Г
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_dense_19721_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Е
AssignVariableOp_41AssignVariableOp-assignvariableop_41_adam_dense_19722_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Г
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_dense_19722_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Е
AssignVariableOp_43AssignVariableOp-assignvariableop_43_adam_dense_19723_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Г
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_dense_19723_bias_vIdentity_44:output:0"/device:CPU:0*
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
ј	
с
H__inference_dense_19721_layer_call_and_return_conditional_losses_4795151

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
Д	

%__inference_signature_wrapper_4795467
conv2d_33801_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_33801_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_47950022
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
_user_specified_nameconv2d_33801_input
п

т
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_4795053

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
Р	

.__inference_sequential_3_layer_call_fn_4795607

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
I__inference_sequential_3_layer_call_and_return_conditional_losses_47953332
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

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795206

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
К
Q
5__inference_max_pooling2d_16906_layer_call_fn_4795026

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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_47950202
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
К
Q
5__inference_max_pooling2d_16907_layer_call_fn_4795038

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
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_47950322
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

G
+__inference_dropout_3_layer_call_fn_4795774

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
F__inference_dropout_3_layer_call_and_return_conditional_losses_47952112
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

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795759

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

l
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_4795008

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
п

т
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_4795647

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
Р4
Ѕ
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795333

inputs
conv2d_33801_4795297
conv2d_33801_4795299
conv2d_33802_4795303
conv2d_33802_4795305
conv2d_33803_4795309
conv2d_33803_4795311
dense_19721_4795316
dense_19721_4795318
dense_19722_4795321
dense_19722_4795323
dense_19723_4795327
dense_19723_4795329
identityЂ$conv2d_33801/StatefulPartitionedCallЂ$conv2d_33802/StatefulPartitionedCallЂ$conv2d_33803/StatefulPartitionedCallЂ#dense_19721/StatefulPartitionedCallЂ#dense_19722/StatefulPartitionedCallЂ#dense_19723/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallИ
$conv2d_33801/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_33801_4795297conv2d_33801_4795299*
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
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_47950532&
$conv2d_33801/StatefulPartitionedCallЊ
#max_pooling2d_16905/PartitionedCallPartitionedCall-conv2d_33801/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_47950082%
#max_pooling2d_16905/PartitionedCallо
$conv2d_33802/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16905/PartitionedCall:output:0conv2d_33802_4795303conv2d_33802_4795305*
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
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_47950812&
$conv2d_33802/StatefulPartitionedCallЈ
#max_pooling2d_16906/PartitionedCallPartitionedCall-conv2d_33802/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_47950202%
#max_pooling2d_16906/PartitionedCallн
$conv2d_33803/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16906/PartitionedCall:output:0conv2d_33803_4795309conv2d_33803_4795311*
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
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_47951092&
$conv2d_33803/StatefulPartitionedCallЉ
#max_pooling2d_16907/PartitionedCallPartitionedCall-conv2d_33803/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_47950322%
#max_pooling2d_16907/PartitionedCall
flatten_5635/PartitionedCallPartitionedCall,max_pooling2d_16907/PartitionedCall:output:0*
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
GPU2*0J 8 *R
fMRK
I__inference_flatten_5635_layer_call_and_return_conditional_losses_47951322
flatten_5635/PartitionedCallШ
#dense_19721/StatefulPartitionedCallStatefulPartitionedCall%flatten_5635/PartitionedCall:output:0dense_19721_4795316dense_19721_4795318*
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
H__inference_dense_19721_layer_call_and_return_conditional_losses_47951512%
#dense_19721/StatefulPartitionedCallЯ
#dense_19722/StatefulPartitionedCallStatefulPartitionedCall,dense_19721/StatefulPartitionedCall:output:0dense_19722_4795321dense_19722_4795323*
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
H__inference_dense_19722_layer_call_and_return_conditional_losses_47951782%
#dense_19722/StatefulPartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall,dense_19722/StatefulPartitionedCall:output:0*
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
F__inference_dropout_3_layer_call_and_return_conditional_losses_47952062#
!dropout_3/StatefulPartitionedCallЭ
#dense_19723/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_19723_4795327dense_19723_4795329*
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
H__inference_dense_19723_layer_call_and_return_conditional_losses_47952352%
#dense_19723/StatefulPartitionedCall
IdentityIdentity,dense_19723/StatefulPartitionedCall:output:0%^conv2d_33801/StatefulPartitionedCall%^conv2d_33802/StatefulPartitionedCall%^conv2d_33803/StatefulPartitionedCall$^dense_19721/StatefulPartitionedCall$^dense_19722/StatefulPartitionedCall$^dense_19723/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_33801/StatefulPartitionedCall$conv2d_33801/StatefulPartitionedCall2L
$conv2d_33802/StatefulPartitionedCall$conv2d_33802/StatefulPartitionedCall2L
$conv2d_33803/StatefulPartitionedCall$conv2d_33803/StatefulPartitionedCall2J
#dense_19721/StatefulPartitionedCall#dense_19721/StatefulPartitionedCall2J
#dense_19722/StatefulPartitionedCall#dense_19722/StatefulPartitionedCall2J
#dense_19723/StatefulPartitionedCall#dense_19723/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_4795032

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


.__inference_conv2d_33803_layer_call_fn_4795696

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
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_47951092
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
й

т
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_4795109

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
ф	
І
.__inference_sequential_3_layer_call_fn_4795428
conv2d_33801_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_33801_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_3_layer_call_and_return_conditional_losses_47954012
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
_user_specified_nameconv2d_33801_input
ь

-__inference_dense_19721_layer_call_fn_4795727

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
H__inference_dense_19721_layer_call_and_return_conditional_losses_47951512
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
њ	
с
H__inference_dense_19723_layer_call_and_return_conditional_losses_4795235

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
Д3

I__inference_sequential_3_layer_call_and_return_conditional_losses_4795291
conv2d_33801_input
conv2d_33801_4795255
conv2d_33801_4795257
conv2d_33802_4795261
conv2d_33802_4795263
conv2d_33803_4795267
conv2d_33803_4795269
dense_19721_4795274
dense_19721_4795276
dense_19722_4795279
dense_19722_4795281
dense_19723_4795285
dense_19723_4795287
identityЂ$conv2d_33801/StatefulPartitionedCallЂ$conv2d_33802/StatefulPartitionedCallЂ$conv2d_33803/StatefulPartitionedCallЂ#dense_19721/StatefulPartitionedCallЂ#dense_19722/StatefulPartitionedCallЂ#dense_19723/StatefulPartitionedCallФ
$conv2d_33801/StatefulPartitionedCallStatefulPartitionedCallconv2d_33801_inputconv2d_33801_4795255conv2d_33801_4795257*
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
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_47950532&
$conv2d_33801/StatefulPartitionedCallЊ
#max_pooling2d_16905/PartitionedCallPartitionedCall-conv2d_33801/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_47950082%
#max_pooling2d_16905/PartitionedCallо
$conv2d_33802/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16905/PartitionedCall:output:0conv2d_33802_4795261conv2d_33802_4795263*
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
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_47950812&
$conv2d_33802/StatefulPartitionedCallЈ
#max_pooling2d_16906/PartitionedCallPartitionedCall-conv2d_33802/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_47950202%
#max_pooling2d_16906/PartitionedCallн
$conv2d_33803/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16906/PartitionedCall:output:0conv2d_33803_4795267conv2d_33803_4795269*
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
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_47951092&
$conv2d_33803/StatefulPartitionedCallЉ
#max_pooling2d_16907/PartitionedCallPartitionedCall-conv2d_33803/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_47950322%
#max_pooling2d_16907/PartitionedCall
flatten_5635/PartitionedCallPartitionedCall,max_pooling2d_16907/PartitionedCall:output:0*
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
GPU2*0J 8 *R
fMRK
I__inference_flatten_5635_layer_call_and_return_conditional_losses_47951322
flatten_5635/PartitionedCallШ
#dense_19721/StatefulPartitionedCallStatefulPartitionedCall%flatten_5635/PartitionedCall:output:0dense_19721_4795274dense_19721_4795276*
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
H__inference_dense_19721_layer_call_and_return_conditional_losses_47951512%
#dense_19721/StatefulPartitionedCallЯ
#dense_19722/StatefulPartitionedCallStatefulPartitionedCall,dense_19721/StatefulPartitionedCall:output:0dense_19722_4795279dense_19722_4795281*
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
H__inference_dense_19722_layer_call_and_return_conditional_losses_47951782%
#dense_19722/StatefulPartitionedCall
dropout_3/PartitionedCallPartitionedCall,dense_19722/StatefulPartitionedCall:output:0*
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
F__inference_dropout_3_layer_call_and_return_conditional_losses_47952112
dropout_3/PartitionedCallХ
#dense_19723/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_19723_4795285dense_19723_4795287*
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
H__inference_dense_19723_layer_call_and_return_conditional_losses_47952352%
#dense_19723/StatefulPartitionedCallч
IdentityIdentity,dense_19723/StatefulPartitionedCall:output:0%^conv2d_33801/StatefulPartitionedCall%^conv2d_33802/StatefulPartitionedCall%^conv2d_33803/StatefulPartitionedCall$^dense_19721/StatefulPartitionedCall$^dense_19722/StatefulPartitionedCall$^dense_19723/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_33801/StatefulPartitionedCall$conv2d_33801/StatefulPartitionedCall2L
$conv2d_33802/StatefulPartitionedCall$conv2d_33802/StatefulPartitionedCall2L
$conv2d_33803/StatefulPartitionedCall$conv2d_33803/StatefulPartitionedCall2J
#dense_19721/StatefulPartitionedCall#dense_19721/StatefulPartitionedCall2J
#dense_19722/StatefulPartitionedCall#dense_19722/StatefulPartitionedCall2J
#dense_19723/StatefulPartitionedCall#dense_19723/StatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_33801_input
Щ
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795211

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
й

т
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_4795687

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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_4795020

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
п

т
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_4795081

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
К
Q
5__inference_max_pooling2d_16905_layer_call_fn_4795014

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
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_47950082
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


.__inference_conv2d_33802_layer_call_fn_4795676

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
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_47950812
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
њ	
с
H__inference_dense_19723_layer_call_and_return_conditional_losses_4795785

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
Р	

.__inference_sequential_3_layer_call_fn_4795636

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
I__inference_sequential_3_layer_call_and_return_conditional_losses_47954012
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
Щ
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795764

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
Н_
Ћ
 __inference__traced_save_4795952
file_prefix2
.savev2_conv2d_33801_kernel_read_readvariableop0
,savev2_conv2d_33801_bias_read_readvariableop2
.savev2_conv2d_33802_kernel_read_readvariableop0
,savev2_conv2d_33802_bias_read_readvariableop2
.savev2_conv2d_33803_kernel_read_readvariableop0
,savev2_conv2d_33803_bias_read_readvariableop1
-savev2_dense_19721_kernel_read_readvariableop/
+savev2_dense_19721_bias_read_readvariableop1
-savev2_dense_19722_kernel_read_readvariableop/
+savev2_dense_19722_bias_read_readvariableop1
-savev2_dense_19723_kernel_read_readvariableop/
+savev2_dense_19723_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_conv2d_33801_kernel_m_read_readvariableop7
3savev2_adam_conv2d_33801_bias_m_read_readvariableop9
5savev2_adam_conv2d_33802_kernel_m_read_readvariableop7
3savev2_adam_conv2d_33802_bias_m_read_readvariableop9
5savev2_adam_conv2d_33803_kernel_m_read_readvariableop7
3savev2_adam_conv2d_33803_bias_m_read_readvariableop8
4savev2_adam_dense_19721_kernel_m_read_readvariableop6
2savev2_adam_dense_19721_bias_m_read_readvariableop8
4savev2_adam_dense_19722_kernel_m_read_readvariableop6
2savev2_adam_dense_19722_bias_m_read_readvariableop8
4savev2_adam_dense_19723_kernel_m_read_readvariableop6
2savev2_adam_dense_19723_bias_m_read_readvariableop9
5savev2_adam_conv2d_33801_kernel_v_read_readvariableop7
3savev2_adam_conv2d_33801_bias_v_read_readvariableop9
5savev2_adam_conv2d_33802_kernel_v_read_readvariableop7
3savev2_adam_conv2d_33802_bias_v_read_readvariableop9
5savev2_adam_conv2d_33803_kernel_v_read_readvariableop7
3savev2_adam_conv2d_33803_bias_v_read_readvariableop8
4savev2_adam_dense_19721_kernel_v_read_readvariableop6
2savev2_adam_dense_19721_bias_v_read_readvariableop8
4savev2_adam_dense_19722_kernel_v_read_readvariableop6
2savev2_adam_dense_19722_bias_v_read_readvariableop8
4savev2_adam_dense_19723_kernel_v_read_readvariableop6
2savev2_adam_dense_19723_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv2d_33801_kernel_read_readvariableop,savev2_conv2d_33801_bias_read_readvariableop.savev2_conv2d_33802_kernel_read_readvariableop,savev2_conv2d_33802_bias_read_readvariableop.savev2_conv2d_33803_kernel_read_readvariableop,savev2_conv2d_33803_bias_read_readvariableop-savev2_dense_19721_kernel_read_readvariableop+savev2_dense_19721_bias_read_readvariableop-savev2_dense_19722_kernel_read_readvariableop+savev2_dense_19722_bias_read_readvariableop-savev2_dense_19723_kernel_read_readvariableop+savev2_dense_19723_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_conv2d_33801_kernel_m_read_readvariableop3savev2_adam_conv2d_33801_bias_m_read_readvariableop5savev2_adam_conv2d_33802_kernel_m_read_readvariableop3savev2_adam_conv2d_33802_bias_m_read_readvariableop5savev2_adam_conv2d_33803_kernel_m_read_readvariableop3savev2_adam_conv2d_33803_bias_m_read_readvariableop4savev2_adam_dense_19721_kernel_m_read_readvariableop2savev2_adam_dense_19721_bias_m_read_readvariableop4savev2_adam_dense_19722_kernel_m_read_readvariableop2savev2_adam_dense_19722_bias_m_read_readvariableop4savev2_adam_dense_19723_kernel_m_read_readvariableop2savev2_adam_dense_19723_bias_m_read_readvariableop5savev2_adam_conv2d_33801_kernel_v_read_readvariableop3savev2_adam_conv2d_33801_bias_v_read_readvariableop5savev2_adam_conv2d_33802_kernel_v_read_readvariableop3savev2_adam_conv2d_33802_bias_v_read_readvariableop5savev2_adam_conv2d_33803_kernel_v_read_readvariableop3savev2_adam_conv2d_33803_bias_v_read_readvariableop4savev2_adam_dense_19721_kernel_v_read_readvariableop2savev2_adam_dense_19721_bias_v_read_readvariableop4savev2_adam_dense_19722_kernel_v_read_readvariableop2savev2_adam_dense_19722_bias_v_read_readvariableop4savev2_adam_dense_19723_kernel_v_read_readvariableop2savev2_adam_dense_19723_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Ѕ
d
+__inference_dropout_3_layer_call_fn_4795769

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
F__inference_dropout_3_layer_call_and_return_conditional_losses_47952062
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
ЌZ

"__inference__wrapped_model_4795002
conv2d_33801_input<
8sequential_3_conv2d_33801_conv2d_readvariableop_resource=
9sequential_3_conv2d_33801_biasadd_readvariableop_resource<
8sequential_3_conv2d_33802_conv2d_readvariableop_resource=
9sequential_3_conv2d_33802_biasadd_readvariableop_resource<
8sequential_3_conv2d_33803_conv2d_readvariableop_resource=
9sequential_3_conv2d_33803_biasadd_readvariableop_resource;
7sequential_3_dense_19721_matmul_readvariableop_resource<
8sequential_3_dense_19721_biasadd_readvariableop_resource;
7sequential_3_dense_19722_matmul_readvariableop_resource<
8sequential_3_dense_19722_biasadd_readvariableop_resource;
7sequential_3_dense_19723_matmul_readvariableop_resource<
8sequential_3_dense_19723_biasadd_readvariableop_resource
identityЂ0sequential_3/conv2d_33801/BiasAdd/ReadVariableOpЂ/sequential_3/conv2d_33801/Conv2D/ReadVariableOpЂ0sequential_3/conv2d_33802/BiasAdd/ReadVariableOpЂ/sequential_3/conv2d_33802/Conv2D/ReadVariableOpЂ0sequential_3/conv2d_33803/BiasAdd/ReadVariableOpЂ/sequential_3/conv2d_33803/Conv2D/ReadVariableOpЂ/sequential_3/dense_19721/BiasAdd/ReadVariableOpЂ.sequential_3/dense_19721/MatMul/ReadVariableOpЂ/sequential_3/dense_19722/BiasAdd/ReadVariableOpЂ.sequential_3/dense_19722/MatMul/ReadVariableOpЂ/sequential_3/dense_19723/BiasAdd/ReadVariableOpЂ.sequential_3/dense_19723/MatMul/ReadVariableOpу
/sequential_3/conv2d_33801/Conv2D/ReadVariableOpReadVariableOp8sequential_3_conv2d_33801_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/sequential_3/conv2d_33801/Conv2D/ReadVariableOpџ
 sequential_3/conv2d_33801/Conv2DConv2Dconv2d_33801_input7sequential_3/conv2d_33801/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2"
 sequential_3/conv2d_33801/Conv2Dк
0sequential_3/conv2d_33801/BiasAdd/ReadVariableOpReadVariableOp9sequential_3_conv2d_33801_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential_3/conv2d_33801/BiasAdd/ReadVariableOpђ
!sequential_3/conv2d_33801/BiasAddBiasAdd)sequential_3/conv2d_33801/Conv2D:output:08sequential_3/conv2d_33801/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2#
!sequential_3/conv2d_33801/BiasAddА
sequential_3/conv2d_33801/ReluRelu*sequential_3/conv2d_33801/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2 
sequential_3/conv2d_33801/Reluќ
(sequential_3/max_pooling2d_16905/MaxPoolMaxPool,sequential_3/conv2d_33801/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2*
(sequential_3/max_pooling2d_16905/MaxPoolу
/sequential_3/conv2d_33802/Conv2D/ReadVariableOpReadVariableOp8sequential_3_conv2d_33802_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/sequential_3/conv2d_33802/Conv2D/ReadVariableOp
 sequential_3/conv2d_33802/Conv2DConv2D1sequential_3/max_pooling2d_16905/MaxPool:output:07sequential_3/conv2d_33802/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2"
 sequential_3/conv2d_33802/Conv2Dк
0sequential_3/conv2d_33802/BiasAdd/ReadVariableOpReadVariableOp9sequential_3_conv2d_33802_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential_3/conv2d_33802/BiasAdd/ReadVariableOpђ
!sequential_3/conv2d_33802/BiasAddBiasAdd)sequential_3/conv2d_33802/Conv2D:output:08sequential_3/conv2d_33802/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2#
!sequential_3/conv2d_33802/BiasAddА
sequential_3/conv2d_33802/ReluRelu*sequential_3/conv2d_33802/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2 
sequential_3/conv2d_33802/Reluњ
(sequential_3/max_pooling2d_16906/MaxPoolMaxPool,sequential_3/conv2d_33802/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@@*
ksize
*
paddingVALID*
strides
2*
(sequential_3/max_pooling2d_16906/MaxPoolф
/sequential_3/conv2d_33803/Conv2D/ReadVariableOpReadVariableOp8sequential_3_conv2d_33803_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype021
/sequential_3/conv2d_33803/Conv2D/ReadVariableOp
 sequential_3/conv2d_33803/Conv2DConv2D1sequential_3/max_pooling2d_16906/MaxPool:output:07sequential_3/conv2d_33803/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2"
 sequential_3/conv2d_33803/Conv2Dл
0sequential_3/conv2d_33803/BiasAdd/ReadVariableOpReadVariableOp9sequential_3_conv2d_33803_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0sequential_3/conv2d_33803/BiasAdd/ReadVariableOpё
!sequential_3/conv2d_33803/BiasAddBiasAdd)sequential_3/conv2d_33803/Conv2D:output:08sequential_3/conv2d_33803/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2#
!sequential_3/conv2d_33803/BiasAddЏ
sequential_3/conv2d_33803/ReluRelu*sequential_3/conv2d_33803/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2 
sequential_3/conv2d_33803/Reluћ
(sequential_3/max_pooling2d_16907/MaxPoolMaxPool,sequential_3/conv2d_33803/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2*
(sequential_3/max_pooling2d_16907/MaxPool
sequential_3/flatten_5635/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2!
sequential_3/flatten_5635/Constт
!sequential_3/flatten_5635/ReshapeReshape1sequential_3/max_pooling2d_16907/MaxPool:output:0(sequential_3/flatten_5635/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2#
!sequential_3/flatten_5635/Reshapeк
.sequential_3/dense_19721/MatMul/ReadVariableOpReadVariableOp7sequential_3_dense_19721_matmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype020
.sequential_3/dense_19721/MatMul/ReadVariableOpт
sequential_3/dense_19721/MatMulMatMul*sequential_3/flatten_5635/Reshape:output:06sequential_3/dense_19721/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2!
sequential_3/dense_19721/MatMulз
/sequential_3/dense_19721/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dense_19721_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype021
/sequential_3/dense_19721/BiasAdd/ReadVariableOpх
 sequential_3/dense_19721/BiasAddBiasAdd)sequential_3/dense_19721/MatMul:product:07sequential_3/dense_19721/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2"
 sequential_3/dense_19721/BiasAddЃ
sequential_3/dense_19721/ReluRelu)sequential_3/dense_19721/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
sequential_3/dense_19721/Reluи
.sequential_3/dense_19722/MatMul/ReadVariableOpReadVariableOp7sequential_3_dense_19722_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype020
.sequential_3/dense_19722/MatMul/ReadVariableOpу
sequential_3/dense_19722/MatMulMatMul+sequential_3/dense_19721/Relu:activations:06sequential_3/dense_19722/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
sequential_3/dense_19722/MatMulз
/sequential_3/dense_19722/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dense_19722_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype021
/sequential_3/dense_19722/BiasAdd/ReadVariableOpх
 sequential_3/dense_19722/BiasAddBiasAdd)sequential_3/dense_19722/MatMul:product:07sequential_3/dense_19722/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 sequential_3/dense_19722/BiasAddЃ
sequential_3/dense_19722/ReluRelu)sequential_3/dense_19722/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
sequential_3/dense_19722/Relu­
sequential_3/dropout_3/IdentityIdentity+sequential_3/dense_19722/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
sequential_3/dropout_3/Identityи
.sequential_3/dense_19723/MatMul/ReadVariableOpReadVariableOp7sequential_3_dense_19723_matmul_readvariableop_resource*
_output_shapes

:<*
dtype020
.sequential_3/dense_19723/MatMul/ReadVariableOpр
sequential_3/dense_19723/MatMulMatMul(sequential_3/dropout_3/Identity:output:06sequential_3/dense_19723/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_3/dense_19723/MatMulз
/sequential_3/dense_19723/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dense_19723_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_3/dense_19723/BiasAdd/ReadVariableOpх
 sequential_3/dense_19723/BiasAddBiasAdd)sequential_3/dense_19723/MatMul:product:07sequential_3/dense_19723/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_3/dense_19723/BiasAddЌ
 sequential_3/dense_19723/SoftmaxSoftmax)sequential_3/dense_19723/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_3/dense_19723/Softmaxж
IdentityIdentity*sequential_3/dense_19723/Softmax:softmax:01^sequential_3/conv2d_33801/BiasAdd/ReadVariableOp0^sequential_3/conv2d_33801/Conv2D/ReadVariableOp1^sequential_3/conv2d_33802/BiasAdd/ReadVariableOp0^sequential_3/conv2d_33802/Conv2D/ReadVariableOp1^sequential_3/conv2d_33803/BiasAdd/ReadVariableOp0^sequential_3/conv2d_33803/Conv2D/ReadVariableOp0^sequential_3/dense_19721/BiasAdd/ReadVariableOp/^sequential_3/dense_19721/MatMul/ReadVariableOp0^sequential_3/dense_19722/BiasAdd/ReadVariableOp/^sequential_3/dense_19722/MatMul/ReadVariableOp0^sequential_3/dense_19723/BiasAdd/ReadVariableOp/^sequential_3/dense_19723/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2d
0sequential_3/conv2d_33801/BiasAdd/ReadVariableOp0sequential_3/conv2d_33801/BiasAdd/ReadVariableOp2b
/sequential_3/conv2d_33801/Conv2D/ReadVariableOp/sequential_3/conv2d_33801/Conv2D/ReadVariableOp2d
0sequential_3/conv2d_33802/BiasAdd/ReadVariableOp0sequential_3/conv2d_33802/BiasAdd/ReadVariableOp2b
/sequential_3/conv2d_33802/Conv2D/ReadVariableOp/sequential_3/conv2d_33802/Conv2D/ReadVariableOp2d
0sequential_3/conv2d_33803/BiasAdd/ReadVariableOp0sequential_3/conv2d_33803/BiasAdd/ReadVariableOp2b
/sequential_3/conv2d_33803/Conv2D/ReadVariableOp/sequential_3/conv2d_33803/Conv2D/ReadVariableOp2b
/sequential_3/dense_19721/BiasAdd/ReadVariableOp/sequential_3/dense_19721/BiasAdd/ReadVariableOp2`
.sequential_3/dense_19721/MatMul/ReadVariableOp.sequential_3/dense_19721/MatMul/ReadVariableOp2b
/sequential_3/dense_19722/BiasAdd/ReadVariableOp/sequential_3/dense_19722/BiasAdd/ReadVariableOp2`
.sequential_3/dense_19722/MatMul/ReadVariableOp.sequential_3/dense_19722/MatMul/ReadVariableOp2b
/sequential_3/dense_19723/BiasAdd/ReadVariableOp/sequential_3/dense_19723/BiasAdd/ReadVariableOp2`
.sequential_3/dense_19723/MatMul/ReadVariableOp.sequential_3/dense_19723/MatMul/ReadVariableOp:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_33801_input
H
№
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795578

inputs/
+conv2d_33801_conv2d_readvariableop_resource0
,conv2d_33801_biasadd_readvariableop_resource/
+conv2d_33802_conv2d_readvariableop_resource0
,conv2d_33802_biasadd_readvariableop_resource/
+conv2d_33803_conv2d_readvariableop_resource0
,conv2d_33803_biasadd_readvariableop_resource.
*dense_19721_matmul_readvariableop_resource/
+dense_19721_biasadd_readvariableop_resource.
*dense_19722_matmul_readvariableop_resource/
+dense_19722_biasadd_readvariableop_resource.
*dense_19723_matmul_readvariableop_resource/
+dense_19723_biasadd_readvariableop_resource
identityЂ#conv2d_33801/BiasAdd/ReadVariableOpЂ"conv2d_33801/Conv2D/ReadVariableOpЂ#conv2d_33802/BiasAdd/ReadVariableOpЂ"conv2d_33802/Conv2D/ReadVariableOpЂ#conv2d_33803/BiasAdd/ReadVariableOpЂ"conv2d_33803/Conv2D/ReadVariableOpЂ"dense_19721/BiasAdd/ReadVariableOpЂ!dense_19721/MatMul/ReadVariableOpЂ"dense_19722/BiasAdd/ReadVariableOpЂ!dense_19722/MatMul/ReadVariableOpЂ"dense_19723/BiasAdd/ReadVariableOpЂ!dense_19723/MatMul/ReadVariableOpМ
"conv2d_33801/Conv2D/ReadVariableOpReadVariableOp+conv2d_33801_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"conv2d_33801/Conv2D/ReadVariableOpЬ
conv2d_33801/Conv2DConv2Dinputs*conv2d_33801/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d_33801/Conv2DГ
#conv2d_33801/BiasAdd/ReadVariableOpReadVariableOp,conv2d_33801_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv2d_33801/BiasAdd/ReadVariableOpО
conv2d_33801/BiasAddBiasAddconv2d_33801/Conv2D:output:0+conv2d_33801/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_33801/BiasAdd
conv2d_33801/ReluReluconv2d_33801/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_33801/Reluе
max_pooling2d_16905/MaxPoolMaxPoolconv2d_33801/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16905/MaxPoolМ
"conv2d_33802/Conv2D/ReadVariableOpReadVariableOp+conv2d_33802_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"conv2d_33802/Conv2D/ReadVariableOpъ
conv2d_33802/Conv2DConv2D$max_pooling2d_16905/MaxPool:output:0*conv2d_33802/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_33802/Conv2DГ
#conv2d_33802/BiasAdd/ReadVariableOpReadVariableOp,conv2d_33802_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv2d_33802/BiasAdd/ReadVariableOpО
conv2d_33802/BiasAddBiasAddconv2d_33802/Conv2D:output:0+conv2d_33802/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_33802/BiasAdd
conv2d_33802/ReluReluconv2d_33802/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_33802/Reluг
max_pooling2d_16906/MaxPoolMaxPoolconv2d_33802/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16906/MaxPoolН
"conv2d_33803/Conv2D/ReadVariableOpReadVariableOp+conv2d_33803_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"conv2d_33803/Conv2D/ReadVariableOpщ
conv2d_33803/Conv2DConv2D$max_pooling2d_16906/MaxPool:output:0*conv2d_33803/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_33803/Conv2DД
#conv2d_33803/BiasAdd/ReadVariableOpReadVariableOp,conv2d_33803_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#conv2d_33803/BiasAdd/ReadVariableOpН
conv2d_33803/BiasAddBiasAddconv2d_33803/Conv2D:output:0+conv2d_33803/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_33803/BiasAdd
conv2d_33803/ReluReluconv2d_33803/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_33803/Reluд
max_pooling2d_16907/MaxPoolMaxPoolconv2d_33803/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16907/MaxPooly
flatten_5635/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2
flatten_5635/ConstЎ
flatten_5635/ReshapeReshape$max_pooling2d_16907/MaxPool:output:0flatten_5635/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2
flatten_5635/ReshapeГ
!dense_19721/MatMul/ReadVariableOpReadVariableOp*dense_19721_matmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype02#
!dense_19721/MatMul/ReadVariableOpЎ
dense_19721/MatMulMatMulflatten_5635/Reshape:output:0)dense_19721/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_19721/MatMulА
"dense_19721/BiasAdd/ReadVariableOpReadVariableOp+dense_19721_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02$
"dense_19721/BiasAdd/ReadVariableOpБ
dense_19721/BiasAddBiasAdddense_19721/MatMul:product:0*dense_19721/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_19721/BiasAdd|
dense_19721/ReluReludense_19721/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_19721/ReluБ
!dense_19722/MatMul/ReadVariableOpReadVariableOp*dense_19722_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02#
!dense_19722/MatMul/ReadVariableOpЏ
dense_19722/MatMulMatMuldense_19721/Relu:activations:0)dense_19722/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_19722/MatMulА
"dense_19722/BiasAdd/ReadVariableOpReadVariableOp+dense_19722_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dense_19722/BiasAdd/ReadVariableOpБ
dense_19722/BiasAddBiasAdddense_19722/MatMul:product:0*dense_19722/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_19722/BiasAdd|
dense_19722/ReluReludense_19722/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_19722/Relu
dropout_3/IdentityIdentitydense_19722/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout_3/IdentityБ
!dense_19723/MatMul/ReadVariableOpReadVariableOp*dense_19723_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!dense_19723/MatMul/ReadVariableOpЌ
dense_19723/MatMulMatMuldropout_3/Identity:output:0)dense_19723/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19723/MatMulА
"dense_19723/BiasAdd/ReadVariableOpReadVariableOp+dense_19723_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_19723/BiasAdd/ReadVariableOpБ
dense_19723/BiasAddBiasAdddense_19723/MatMul:product:0*dense_19723/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19723/BiasAdd
dense_19723/SoftmaxSoftmaxdense_19723/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19723/Softmax­
IdentityIdentitydense_19723/Softmax:softmax:0$^conv2d_33801/BiasAdd/ReadVariableOp#^conv2d_33801/Conv2D/ReadVariableOp$^conv2d_33802/BiasAdd/ReadVariableOp#^conv2d_33802/Conv2D/ReadVariableOp$^conv2d_33803/BiasAdd/ReadVariableOp#^conv2d_33803/Conv2D/ReadVariableOp#^dense_19721/BiasAdd/ReadVariableOp"^dense_19721/MatMul/ReadVariableOp#^dense_19722/BiasAdd/ReadVariableOp"^dense_19722/MatMul/ReadVariableOp#^dense_19723/BiasAdd/ReadVariableOp"^dense_19723/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2J
#conv2d_33801/BiasAdd/ReadVariableOp#conv2d_33801/BiasAdd/ReadVariableOp2H
"conv2d_33801/Conv2D/ReadVariableOp"conv2d_33801/Conv2D/ReadVariableOp2J
#conv2d_33802/BiasAdd/ReadVariableOp#conv2d_33802/BiasAdd/ReadVariableOp2H
"conv2d_33802/Conv2D/ReadVariableOp"conv2d_33802/Conv2D/ReadVariableOp2J
#conv2d_33803/BiasAdd/ReadVariableOp#conv2d_33803/BiasAdd/ReadVariableOp2H
"conv2d_33803/Conv2D/ReadVariableOp"conv2d_33803/Conv2D/ReadVariableOp2H
"dense_19721/BiasAdd/ReadVariableOp"dense_19721/BiasAdd/ReadVariableOp2F
!dense_19721/MatMul/ReadVariableOp!dense_19721/MatMul/ReadVariableOp2H
"dense_19722/BiasAdd/ReadVariableOp"dense_19722/BiasAdd/ReadVariableOp2F
!dense_19722/MatMul/ReadVariableOp!dense_19722/MatMul/ReadVariableOp2H
"dense_19723/BiasAdd/ReadVariableOp"dense_19723/BiasAdd/ReadVariableOp2F
!dense_19723/MatMul/ReadVariableOp!dense_19723/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п

т
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_4795667

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
ј	
с
H__inference_dense_19721_layer_call_and_return_conditional_losses_4795718

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
H__inference_dense_19722_layer_call_and_return_conditional_losses_4795178

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
-__inference_dense_19723_layer_call_fn_4795794

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
H__inference_dense_19723_layer_call_and_return_conditional_losses_47952352
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
 
_user_specified_nameinputs
ЖQ
№
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795526

inputs/
+conv2d_33801_conv2d_readvariableop_resource0
,conv2d_33801_biasadd_readvariableop_resource/
+conv2d_33802_conv2d_readvariableop_resource0
,conv2d_33802_biasadd_readvariableop_resource/
+conv2d_33803_conv2d_readvariableop_resource0
,conv2d_33803_biasadd_readvariableop_resource.
*dense_19721_matmul_readvariableop_resource/
+dense_19721_biasadd_readvariableop_resource.
*dense_19722_matmul_readvariableop_resource/
+dense_19722_biasadd_readvariableop_resource.
*dense_19723_matmul_readvariableop_resource/
+dense_19723_biasadd_readvariableop_resource
identityЂ#conv2d_33801/BiasAdd/ReadVariableOpЂ"conv2d_33801/Conv2D/ReadVariableOpЂ#conv2d_33802/BiasAdd/ReadVariableOpЂ"conv2d_33802/Conv2D/ReadVariableOpЂ#conv2d_33803/BiasAdd/ReadVariableOpЂ"conv2d_33803/Conv2D/ReadVariableOpЂ"dense_19721/BiasAdd/ReadVariableOpЂ!dense_19721/MatMul/ReadVariableOpЂ"dense_19722/BiasAdd/ReadVariableOpЂ!dense_19722/MatMul/ReadVariableOpЂ"dense_19723/BiasAdd/ReadVariableOpЂ!dense_19723/MatMul/ReadVariableOpМ
"conv2d_33801/Conv2D/ReadVariableOpReadVariableOp+conv2d_33801_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"conv2d_33801/Conv2D/ReadVariableOpЬ
conv2d_33801/Conv2DConv2Dinputs*conv2d_33801/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d_33801/Conv2DГ
#conv2d_33801/BiasAdd/ReadVariableOpReadVariableOp,conv2d_33801_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv2d_33801/BiasAdd/ReadVariableOpО
conv2d_33801/BiasAddBiasAddconv2d_33801/Conv2D:output:0+conv2d_33801/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_33801/BiasAdd
conv2d_33801/ReluReluconv2d_33801/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ 2
conv2d_33801/Reluе
max_pooling2d_16905/MaxPoolMaxPoolconv2d_33801/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16905/MaxPoolМ
"conv2d_33802/Conv2D/ReadVariableOpReadVariableOp+conv2d_33802_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"conv2d_33802/Conv2D/ReadVariableOpъ
conv2d_33802/Conv2DConv2D$max_pooling2d_16905/MaxPool:output:0*conv2d_33802/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_33802/Conv2DГ
#conv2d_33802/BiasAdd/ReadVariableOpReadVariableOp,conv2d_33802_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv2d_33802/BiasAdd/ReadVariableOpО
conv2d_33802/BiasAddBiasAddconv2d_33802/Conv2D:output:0+conv2d_33802/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_33802/BiasAdd
conv2d_33802/ReluReluconv2d_33802/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ@2
conv2d_33802/Reluг
max_pooling2d_16906/MaxPoolMaxPoolconv2d_33802/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16906/MaxPoolН
"conv2d_33803/Conv2D/ReadVariableOpReadVariableOp+conv2d_33803_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"conv2d_33803/Conv2D/ReadVariableOpщ
conv2d_33803/Conv2DConv2D$max_pooling2d_16906/MaxPool:output:0*conv2d_33803/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_33803/Conv2DД
#conv2d_33803/BiasAdd/ReadVariableOpReadVariableOp,conv2d_33803_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#conv2d_33803/BiasAdd/ReadVariableOpН
conv2d_33803/BiasAddBiasAddconv2d_33803/Conv2D:output:0+conv2d_33803/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_33803/BiasAdd
conv2d_33803/ReluReluconv2d_33803/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@2
conv2d_33803/Reluд
max_pooling2d_16907/MaxPoolMaxPoolconv2d_33803/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16907/MaxPooly
flatten_5635/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџм  2
flatten_5635/ConstЎ
flatten_5635/ReshapeReshape$max_pooling2d_16907/MaxPool:output:0flatten_5635/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџЙ2
flatten_5635/ReshapeГ
!dense_19721/MatMul/ReadVariableOpReadVariableOp*dense_19721_matmul_readvariableop_resource* 
_output_shapes
:
Йx*
dtype02#
!dense_19721/MatMul/ReadVariableOpЎ
dense_19721/MatMulMatMulflatten_5635/Reshape:output:0)dense_19721/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_19721/MatMulА
"dense_19721/BiasAdd/ReadVariableOpReadVariableOp+dense_19721_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02$
"dense_19721/BiasAdd/ReadVariableOpБ
dense_19721/BiasAddBiasAdddense_19721/MatMul:product:0*dense_19721/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_19721/BiasAdd|
dense_19721/ReluReludense_19721/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
dense_19721/ReluБ
!dense_19722/MatMul/ReadVariableOpReadVariableOp*dense_19722_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02#
!dense_19722/MatMul/ReadVariableOpЏ
dense_19722/MatMulMatMuldense_19721/Relu:activations:0)dense_19722/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_19722/MatMulА
"dense_19722/BiasAdd/ReadVariableOpReadVariableOp+dense_19722_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02$
"dense_19722/BiasAdd/ReadVariableOpБ
dense_19722/BiasAddBiasAdddense_19722/MatMul:product:0*dense_19722/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_19722/BiasAdd|
dense_19722/ReluReludense_19722/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dense_19722/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/ConstЉ
dropout_3/dropout/MulMuldense_19722/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeShapedense_19722/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeв
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yц
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ<2
dropout_3/dropout/CastЂ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
dropout_3/dropout/Mul_1Б
!dense_19723/MatMul/ReadVariableOpReadVariableOp*dense_19723_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!dense_19723/MatMul/ReadVariableOpЌ
dense_19723/MatMulMatMuldropout_3/dropout/Mul_1:z:0)dense_19723/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19723/MatMulА
"dense_19723/BiasAdd/ReadVariableOpReadVariableOp+dense_19723_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_19723/BiasAdd/ReadVariableOpБ
dense_19723/BiasAddBiasAdddense_19723/MatMul:product:0*dense_19723/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19723/BiasAdd
dense_19723/SoftmaxSoftmaxdense_19723/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_19723/Softmax­
IdentityIdentitydense_19723/Softmax:softmax:0$^conv2d_33801/BiasAdd/ReadVariableOp#^conv2d_33801/Conv2D/ReadVariableOp$^conv2d_33802/BiasAdd/ReadVariableOp#^conv2d_33802/Conv2D/ReadVariableOp$^conv2d_33803/BiasAdd/ReadVariableOp#^conv2d_33803/Conv2D/ReadVariableOp#^dense_19721/BiasAdd/ReadVariableOp"^dense_19721/MatMul/ReadVariableOp#^dense_19722/BiasAdd/ReadVariableOp"^dense_19722/MatMul/ReadVariableOp#^dense_19723/BiasAdd/ReadVariableOp"^dense_19723/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2J
#conv2d_33801/BiasAdd/ReadVariableOp#conv2d_33801/BiasAdd/ReadVariableOp2H
"conv2d_33801/Conv2D/ReadVariableOp"conv2d_33801/Conv2D/ReadVariableOp2J
#conv2d_33802/BiasAdd/ReadVariableOp#conv2d_33802/BiasAdd/ReadVariableOp2H
"conv2d_33802/Conv2D/ReadVariableOp"conv2d_33802/Conv2D/ReadVariableOp2J
#conv2d_33803/BiasAdd/ReadVariableOp#conv2d_33803/BiasAdd/ReadVariableOp2H
"conv2d_33803/Conv2D/ReadVariableOp"conv2d_33803/Conv2D/ReadVariableOp2H
"dense_19721/BiasAdd/ReadVariableOp"dense_19721/BiasAdd/ReadVariableOp2F
!dense_19721/MatMul/ReadVariableOp!dense_19721/MatMul/ReadVariableOp2H
"dense_19722/BiasAdd/ReadVariableOp"dense_19722/BiasAdd/ReadVariableOp2F
!dense_19722/MatMul/ReadVariableOp!dense_19722/MatMul/ReadVariableOp2H
"dense_19723/BiasAdd/ReadVariableOp"dense_19723/BiasAdd/ReadVariableOp2F
!dense_19723/MatMul/ReadVariableOp!dense_19723/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
J
.__inference_flatten_5635_layer_call_fn_4795707

inputs
identityЬ
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
GPU2*0J 8 *R
fMRK
I__inference_flatten_5635_layer_call_and_return_conditional_losses_47951322
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
Х
e
I__inference_flatten_5635_layer_call_and_return_conditional_losses_4795702

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
ф4
Б
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795252
conv2d_33801_input
conv2d_33801_4795064
conv2d_33801_4795066
conv2d_33802_4795092
conv2d_33802_4795094
conv2d_33803_4795120
conv2d_33803_4795122
dense_19721_4795162
dense_19721_4795164
dense_19722_4795189
dense_19722_4795191
dense_19723_4795246
dense_19723_4795248
identityЂ$conv2d_33801/StatefulPartitionedCallЂ$conv2d_33802/StatefulPartitionedCallЂ$conv2d_33803/StatefulPartitionedCallЂ#dense_19721/StatefulPartitionedCallЂ#dense_19722/StatefulPartitionedCallЂ#dense_19723/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallФ
$conv2d_33801/StatefulPartitionedCallStatefulPartitionedCallconv2d_33801_inputconv2d_33801_4795064conv2d_33801_4795066*
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
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_47950532&
$conv2d_33801/StatefulPartitionedCallЊ
#max_pooling2d_16905/PartitionedCallPartitionedCall-conv2d_33801/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_47950082%
#max_pooling2d_16905/PartitionedCallо
$conv2d_33802/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16905/PartitionedCall:output:0conv2d_33802_4795092conv2d_33802_4795094*
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
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_47950812&
$conv2d_33802/StatefulPartitionedCallЈ
#max_pooling2d_16906/PartitionedCallPartitionedCall-conv2d_33802/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_47950202%
#max_pooling2d_16906/PartitionedCallн
$conv2d_33803/StatefulPartitionedCallStatefulPartitionedCall,max_pooling2d_16906/PartitionedCall:output:0conv2d_33803_4795120conv2d_33803_4795122*
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
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_47951092&
$conv2d_33803/StatefulPartitionedCallЉ
#max_pooling2d_16907/PartitionedCallPartitionedCall-conv2d_33803/StatefulPartitionedCall:output:0*
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
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_47950322%
#max_pooling2d_16907/PartitionedCall
flatten_5635/PartitionedCallPartitionedCall,max_pooling2d_16907/PartitionedCall:output:0*
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
GPU2*0J 8 *R
fMRK
I__inference_flatten_5635_layer_call_and_return_conditional_losses_47951322
flatten_5635/PartitionedCallШ
#dense_19721/StatefulPartitionedCallStatefulPartitionedCall%flatten_5635/PartitionedCall:output:0dense_19721_4795162dense_19721_4795164*
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
H__inference_dense_19721_layer_call_and_return_conditional_losses_47951512%
#dense_19721/StatefulPartitionedCallЯ
#dense_19722/StatefulPartitionedCallStatefulPartitionedCall,dense_19721/StatefulPartitionedCall:output:0dense_19722_4795189dense_19722_4795191*
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
H__inference_dense_19722_layer_call_and_return_conditional_losses_47951782%
#dense_19722/StatefulPartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall,dense_19722/StatefulPartitionedCall:output:0*
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
F__inference_dropout_3_layer_call_and_return_conditional_losses_47952062#
!dropout_3/StatefulPartitionedCallЭ
#dense_19723/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_19723_4795246dense_19723_4795248*
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
H__inference_dense_19723_layer_call_and_return_conditional_losses_47952352%
#dense_19723/StatefulPartitionedCall
IdentityIdentity,dense_19723/StatefulPartitionedCall:output:0%^conv2d_33801/StatefulPartitionedCall%^conv2d_33802/StatefulPartitionedCall%^conv2d_33803/StatefulPartitionedCall$^dense_19721/StatefulPartitionedCall$^dense_19722/StatefulPartitionedCall$^dense_19723/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџ::::::::::::2L
$conv2d_33801/StatefulPartitionedCall$conv2d_33801/StatefulPartitionedCall2L
$conv2d_33802/StatefulPartitionedCall$conv2d_33802/StatefulPartitionedCall2L
$conv2d_33803/StatefulPartitionedCall$conv2d_33803/StatefulPartitionedCall2J
#dense_19721/StatefulPartitionedCall#dense_19721/StatefulPartitionedCall2J
#dense_19722/StatefulPartitionedCall#dense_19722/StatefulPartitionedCall2J
#dense_19723/StatefulPartitionedCall#dense_19723/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameconv2d_33801_input"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ю
serving_defaultК
[
conv2d_33801_inputE
$serving_default_conv2d_33801_input:0џџџџџџџџџ?
dense_197230
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:го
йV
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
А_default_save_signature"пR
_tf_keras_sequentialРR{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_33801_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33801", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16905", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33802", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16906", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33803", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16907", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_5635", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_19721", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19722", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_19723", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_33801_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33801", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16905", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33802", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16906", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33803", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16907", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_5635", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_19721", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19722", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_19723", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"к	
_tf_keras_layerР	{"class_name": "Conv2D", "name": "conv2d_33801", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_33801", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}

regularization_losses
	variables
trainable_variables
	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d_16905", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_16905", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ў	

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Conv2D", "name": "conv2d_33802", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_33802", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}

"regularization_losses
#	variables
$trainable_variables
%	keras_api
З__call__
+И&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d_16906", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_16906", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
§	

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Conv2D", "name": "conv2d_33803", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_33803", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

,regularization_losses
-	variables
.trainable_variables
/	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d_16907", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_16907", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю
0regularization_losses
1	variables
2trainable_variables
3	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"н
_tf_keras_layerУ{"class_name": "Flatten", "name": "flatten_5635", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5635", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"к
_tf_keras_layerР{"class_name": "Dense", "name": "dense_19721", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19721", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 56448}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56448]}}
ќ

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"class_name": "Dense", "name": "dense_19722", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19722", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
ч
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ќ

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"class_name": "Dense", "name": "dense_19723", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19723", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
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
-:+ 2conv2d_33801/kernel
: 2conv2d_33801/bias
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
-:+ @2conv2d_33802/kernel
:@2conv2d_33802/bias
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
.:,@2conv2d_33803/kernel
 :2conv2d_33803/bias
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
Йx2dense_19721/kernel
:x2dense_19721/bias
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
$:"x<2dense_19722/kernel
:<2dense_19722/bias
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
$:"<2dense_19723/kernel
:2dense_19723/bias
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
2:0 2Adam/conv2d_33801/kernel/m
$:" 2Adam/conv2d_33801/bias/m
2:0 @2Adam/conv2d_33802/kernel/m
$:"@2Adam/conv2d_33802/bias/m
3:1@2Adam/conv2d_33803/kernel/m
%:#2Adam/conv2d_33803/bias/m
+:)
Йx2Adam/dense_19721/kernel/m
#:!x2Adam/dense_19721/bias/m
):'x<2Adam/dense_19722/kernel/m
#:!<2Adam/dense_19722/bias/m
):'<2Adam/dense_19723/kernel/m
#:!2Adam/dense_19723/bias/m
2:0 2Adam/conv2d_33801/kernel/v
$:" 2Adam/conv2d_33801/bias/v
2:0 @2Adam/conv2d_33802/kernel/v
$:"@2Adam/conv2d_33802/bias/v
3:1@2Adam/conv2d_33803/kernel/v
%:#2Adam/conv2d_33803/bias/v
+:)
Йx2Adam/dense_19721/kernel/v
#:!x2Adam/dense_19721/bias/v
):'x<2Adam/dense_19722/kernel/v
#:!<2Adam/dense_19722/bias/v
):'<2Adam/dense_19723/kernel/v
#:!2Adam/dense_19723/bias/v
2
.__inference_sequential_3_layer_call_fn_4795428
.__inference_sequential_3_layer_call_fn_4795360
.__inference_sequential_3_layer_call_fn_4795607
.__inference_sequential_3_layer_call_fn_4795636Р
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
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795252
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795526
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795291
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795578Р
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
"__inference__wrapped_model_4795002Ы
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
conv2d_33801_inputџџџџџџџџџ
и2е
.__inference_conv2d_33801_layer_call_fn_4795656Ђ
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
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_4795647Ђ
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
5__inference_max_pooling2d_16905_layer_call_fn_4795014р
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
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_4795008р
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
.__inference_conv2d_33802_layer_call_fn_4795676Ђ
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
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_4795667Ђ
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
5__inference_max_pooling2d_16906_layer_call_fn_4795026р
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
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_4795020р
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
.__inference_conv2d_33803_layer_call_fn_4795696Ђ
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
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_4795687Ђ
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
5__inference_max_pooling2d_16907_layer_call_fn_4795038р
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
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_4795032р
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
.__inference_flatten_5635_layer_call_fn_4795707Ђ
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
I__inference_flatten_5635_layer_call_and_return_conditional_losses_4795702Ђ
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
-__inference_dense_19721_layer_call_fn_4795727Ђ
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
H__inference_dense_19721_layer_call_and_return_conditional_losses_4795718Ђ
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
-__inference_dense_19722_layer_call_fn_4795747Ђ
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
H__inference_dense_19722_layer_call_and_return_conditional_losses_4795738Ђ
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
+__inference_dropout_3_layer_call_fn_4795769
+__inference_dropout_3_layer_call_fn_4795774Д
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
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795759
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795764Д
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
-__inference_dense_19723_layer_call_fn_4795794Ђ
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
H__inference_dense_19723_layer_call_and_return_conditional_losses_4795785Ђ
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
%__inference_signature_wrapper_4795467conv2d_33801_input"
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
"__inference__wrapped_model_4795002&'45:;DEEЂB
;Ђ8
63
conv2d_33801_inputџџџџџџџџџ
Њ "9Њ6
4
dense_19723%"
dense_19723џџџџџџџџџН
I__inference_conv2d_33801_layer_call_and_return_conditional_losses_4795647p9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ 
 
.__inference_conv2d_33801_layer_call_fn_4795656c9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџ Н
I__inference_conv2d_33802_layer_call_and_return_conditional_losses_4795667p9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ 
Њ "/Ђ,
%"
0џџџџџџџџџ@
 
.__inference_conv2d_33802_layer_call_fn_4795676c9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ 
Њ ""џџџџџџџџџ@К
I__inference_conv2d_33803_layer_call_and_return_conditional_losses_4795687m&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@@
Њ ".Ђ+
$!
0џџџџџџџџџ@@
 
.__inference_conv2d_33803_layer_call_fn_4795696`&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@@
Њ "!џџџџџџџџџ@@Њ
H__inference_dense_19721_layer_call_and_return_conditional_losses_4795718^451Ђ.
'Ђ$
"
inputsџџџџџџџџџЙ
Њ "%Ђ"

0џџџџџџџџџx
 
-__inference_dense_19721_layer_call_fn_4795727Q451Ђ.
'Ђ$
"
inputsџџџџџџџџџЙ
Њ "џџџџџџџџџxЈ
H__inference_dense_19722_layer_call_and_return_conditional_losses_4795738\:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "%Ђ"

0џџџџџџџџџ<
 
-__inference_dense_19722_layer_call_fn_4795747O:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "џџџџџџџџџ<Ј
H__inference_dense_19723_layer_call_and_return_conditional_losses_4795785\DE/Ђ,
%Ђ"
 
inputsџџџџџџџџџ<
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_dense_19723_layer_call_fn_4795794ODE/Ђ,
%Ђ"
 
inputsџџџџџџџџџ<
Њ "џџџџџџџџџІ
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795759\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p
Њ "%Ђ"

0џџџџџџџџџ<
 І
F__inference_dropout_3_layer_call_and_return_conditional_losses_4795764\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p 
Њ "%Ђ"

0џџџџџџџџџ<
 ~
+__inference_dropout_3_layer_call_fn_4795769O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p
Њ "џџџџџџџџџ<~
+__inference_dropout_3_layer_call_fn_4795774O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ<
p 
Њ "џџџџџџџџџ<А
I__inference_flatten_5635_layer_call_and_return_conditional_losses_4795702c8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "'Ђ$

0џџџџџџџџџЙ
 
.__inference_flatten_5635_layer_call_fn_4795707V8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "џџџџџџџџџЙѓ
P__inference_max_pooling2d_16905_layer_call_and_return_conditional_losses_4795008RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
5__inference_max_pooling2d_16905_layer_call_fn_4795014RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
P__inference_max_pooling2d_16906_layer_call_and_return_conditional_losses_4795020RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
5__inference_max_pooling2d_16906_layer_call_fn_4795026RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
P__inference_max_pooling2d_16907_layer_call_and_return_conditional_losses_4795032RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
5__inference_max_pooling2d_16907_layer_call_fn_4795038RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџв
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795252&'45:;DEMЂJ
CЂ@
63
conv2d_33801_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 в
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795291&'45:;DEMЂJ
CЂ@
63
conv2d_33801_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795526x&'45:;DEAЂ>
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
I__inference_sequential_3_layer_call_and_return_conditional_losses_4795578x&'45:;DEAЂ>
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
.__inference_sequential_3_layer_call_fn_4795360w&'45:;DEMЂJ
CЂ@
63
conv2d_33801_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџЉ
.__inference_sequential_3_layer_call_fn_4795428w&'45:;DEMЂJ
CЂ@
63
conv2d_33801_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_3_layer_call_fn_4795607k&'45:;DEAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_3_layer_call_fn_4795636k&'45:;DEAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџа
%__inference_signature_wrapper_4795467І&'45:;DE[ЂX
Ђ 
QЊN
L
conv2d_33801_input63
conv2d_33801_inputџџџџџџџџџ"9Њ6
4
dense_19723%"
dense_19723џџџџџџџџџ