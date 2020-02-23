# This file is a part of Julia. License is MIT: https://julialang.org/license

module XoshiroSimd
# Getting the xoroshiro RNG to reliably vectorize is somewhat of a hassle without Simd.jl.
import ..Random: TaskLocalRNG, rand, rand!, Xoshiro, CloseOpen01, UnsafeView,
                 SamplerType, SamplerTrivial
using Base: BitInteger_types

# Vector-width. Influences random stream. We may want to tune this before merging.
xoshiroWidth() = Val(8)
# Simd threshold. Influences random stream. We may want to tune this before merging.
simdThreshold(::Type{T}) where T = 2048
simdThreshold(::Type{Bool}) = 4096

@inline _rotl45(x::UInt64) = (x<<45)|(x>>19)
@inline _shl17(x::UInt64) = x<<17
@inline _rotl23(x::UInt64) = (x<<23)|(x>>41)
@inline _plus(x::UInt64,y::UInt64) = x+y
@inline _xor(x::UInt64,y::UInt64) = xor(x,y)
@inline _and(x::UInt64, y::UInt64) = x & y
@inline _or(x::UInt64, y::UInt64) = x | y
@inline _lshr(x, y::Int32) = _lshr(x, y % Int64)
@inline _lshr(x::UInt64, y::Int64) = Core.Intrinsics.llvmcall("""
    %res = lshr i64 %0, %1
    ret i64 %res
    """,
    UInt64,
    Tuple{UInt64, Int64},
    x, y)

@inline _minus1(x::UInt64, ::Type{Float64}) = reinterpret(UInt64, reinterpret(Float64, x) - 1.0)
@inline function _minus1(x::UInt64, ::Type{Float32})
    u = reinterpret(Float32, (x>>32) % UInt32) - 1.0f0
    l = reinterpret(Float32, x % UInt32) - 1.0f0
    (UInt64(reinterpret(UInt32, u)) << 32) | UInt64(reinterpret(UInt32, l))
end

# required operations. These could be written more concisely with `ntuple`, but the compiler
# sometimes refuses to properly vectorize.
for N in [4,8,16]
    let code, s, fshl = "llvm.fshl.v$(N)i64",
        VT = :(NTuple{$N, VecElement{UInt64}})

        s = ntuple(_->VecElement(UInt64(45)), N)
        @eval @inline _rotl45(x::$VT) = ccall($fshl, llvmcall, $VT, ($VT, $VT, $VT), x, x, $s)

        s = ntuple(_->VecElement(UInt64(23)), N)
        @eval @inline _rotl23(x::$VT) = ccall($fshl, llvmcall, $VT, ($VT, $VT, $VT), x, x, $s)

        code = """
        %lshiftOp = shufflevector <1 x i64> <i64 17>, <1 x i64> undef, <$N x i32> zeroinitializer
        %res = shl <$N x i64> %0, %lshiftOp
        ret <$N x i64> %res
        """
        @eval @inline _shl17(x::$VT) = Core.Intrinsics.llvmcall($code,
            $VT, Tuple{$VT}, x)

        code = """
        %res = add <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _plus(x::$VT, y::$VT) = Core.Intrinsics.llvmcall($code,
            $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %res = xor <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _xor(x::$VT, y::$VT) = Core.Intrinsics.llvmcall($code,
            $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %res = and <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _and(x::$VT, y::$VT) = Core.Intrinsics.llvmcall($code,
            $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %res = or <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _or(x::$VT, y::$VT) = Core.Intrinsics.llvmcall($code,
            $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %tmp = insertelement <1 x i64> undef, i64 %1, i32 0
        %shift = shufflevector <1 x i64> %tmp, <1 x i64> %tmp, <$N x i32> zeroinitializer
        %res = lshr <$N x i64> %0, %shift
        ret <$N x i64> %res
        """
        @eval @inline _lshr(x::$VT, y::Int64) = Core.Intrinsics.llvmcall($code,
            $VT, Tuple{$VT, Int64}, x, y)

        s = "<" * join(fill("double 1.0", N), ", ") * ">"
        code = """
        %f = bitcast <$N x i64> %0 to <$N x double>
        %res = fsub <$N x double> %f, $s
        %i = bitcast <$N x double> %res to <$N x i64>
        ret <$N x i64> %i
        """
        @eval @inline function _minus1(x::$VT, ::Type{Float64})
            Core.Intrinsics.llvmcall($code, $VT, Tuple{$VT}, x)
        end

        s = "<" * join(fill("float 1.0", 2N), ", ") * ">"
        code = """
        %f = bitcast <$N x i64> %0 to <$(2N) x float>
        %res = fsub <$(2N) x float> %f, $s
        %i = bitcast <$(2N) x float> %res to <$N x i64>
        ret <$N x i64> %i
        """
        @eval @inline function _minus1(x::$VT, ::Type{Float32})
            Core.Intrinsics.llvmcall($code, $VT, Tuple{$VT}, x)
        end
    end
end


# The mast values are used as res = _or(_and(res, mskBits), oneBits)

mskBits(::Type{T}) where T = 0xffffffffffffffff
mskBits(::Type{T}, ::Val{N}) where {T,N} = ntuple(i->VecElement(mskBits(T)), Val(N))

oneBits(::Type{T}) where T = 0x0000000000000000
oneBits(::Type{T}, ::Val{N}) where {T,N} = ntuple(i->VecElement(oneBits(T)), Val(N))

mskBits(::Type{Float64}) = Base.significand_mask(Float64)
oneBits(::Type{Float64}) = Base.exponent_one(Float64)

mskBits(::Type{Float32}) = Base.significand_mask(Float32) + UInt64(Base.significand_mask(Float32))<<32
oneBits(::Type{Float32}) = Base.exponent_one(Float32) + UInt64(Base.exponent_one(Float32))<<32


function forkRand(rng::Union{TaskLocalRNG, Xoshiro}, ::Val{N}) where N
    # constants have nothing up their sleeve. For more discussion, cf rng_split in task.c
    # 0x02011ce34bce797f == hash(UInt(1))|0x01
    # 0x5a94851fb48a6e05 == hash(UInt(2))|0x01
    # 0x3688cf5d48899fa7 == hash(UInt(3))|0x01
    # 0x867b4bb4c42e5661 == hash(UInt(4))|0x01
    s0 = ntuple(i->VecElement(0x02011ce34bce797f * rand(rng, UInt64)), Val(N))
    s1 = ntuple(i->VecElement(0x5a94851fb48a6e05 * rand(rng, UInt64)), Val(N))
    s2 = ntuple(i->VecElement(0x3688cf5d48899fa7 * rand(rng, UInt64)), Val(N))
    s3 = ntuple(i->VecElement(0x867b4bb4c42e5661 * rand(rng, UInt64)), Val(N))
    (s0, s1, s2, s3)
end

_id(x, T) = x

@inline function xoshiro_bulk(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, T::Union{Type{UInt8}, Type{Bool}, Type{Float32}, Type{Float64}}, ::Val{N}, f::F = _id) where {N, F}
    if len >= simdThreshold(T)
        written = xoshiro_bulk_simd(rng, dst, len, T, Val(N), f)
        len -= written
        dst += written
    end
    if len != 0
        xoshiro_bulk_nosimd(rng, dst, len, T, f)
    end
    nothing
end

@noinline function xoshiro_bulk_nosimd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{T}, f::F) where {T, F}
    # we rely on specialization, llvm const-prop + instcombine to remove unneeded masking
    mskOR =  oneBits(T)
    mskAND = mskBits(T)
    if rng isa TaskLocalRNG
        task = current_task()
        s0, s1, s2, s3 = task.rngState0, task.rngState1, task.rngState2, task.rngState3
    else
        (; s0, s1, s2, s3) = rng::Xoshiro
    end

    i = 0
    while i+8 <= len
        res = _rotl23(_plus(s0,s3))
        res = _or(_and(res, mskAND), mskOR)
        unsafe_store!(reinterpret(Ptr{UInt64}, dst + i), f(res, T))
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        i += 8
    end
    if i < len
        res = _rotl23(_plus(s0,s3))
        res = _or(_and(res, mskAND), mskOR)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        ref = Ref(f(res, T))
        # TODO: This may make the random-stream dependent on system endianness
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{UInt8}, Ptr{UInt64}, Csize_t), dst+i, ref, len-i)
    end
    if rng isa TaskLocalRNG
        task.rngState0, task.rngState1, task.rngState2, task.rngState3 = s0, s1, s2, s3
    else
       rng.s0, rng.s1, rng.s2, rng.s3 =  s0, s1, s2, s3
    end
    nothing
end

@noinline function xoshiro_bulk_nosimd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{Bool}, f)
    if rng isa TaskLocalRNG
        task = current_task()
        s0, s1, s2, s3 = task.rngState0, task.rngState1, task.rngState2, task.rngState3
    else
        (; s0, s1, s2, s3) = rng::Xoshiro
    end

    i = 0
    while i+8 <= len
        res = _rotl23(_plus(s0,s3))
        shift = 0
        while i+8 <= len && shift < 8
            resLoc = _and(_lshr(res, shift), 0x0101010101010101)
            unsafe_store!(reinterpret(Ptr{UInt64}, dst + i), resLoc)
            i += 8
            shift += 1
        end

        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
    end
    if i < len
        # we may overgenerate some bytes here, if len mod 64 <= 56 and len mod 8 != 0
        res = _rotl23(_plus(s0,s3))
        resLoc = _and(res, 0x0101010101010101)
        ref = Ref(resLoc)
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{UInt8}, Ptr{UInt64}, Csize_t), dst+i, ref, len-i)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
    end
    if rng isa TaskLocalRNG
        task.rngState0, task.rngState1, task.rngState2, task.rngState3 = s0, s1, s2, s3
    else
        rng.s0, rng.s1, rng.s2, rng.s3 = s0, s1, s2, s3
    end
    nothing
end


@noinline function xoshiro_bulk_simd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{T}, ::Val{N}, f::F) where {T,N,F}
    # we rely on specialization, llvm const-prop + instcombine to remove unneeded masking
    mskOR =  oneBits(T, Val(N))
    mskAND = mskBits(T, Val(N))

    s0, s1, s2, s3 = forkRand(rng, Val(N))

    i = 0
    while i + 8*N <= len
        res = _rotl23(_plus(s0,s3))
        res = _or(_and(res, mskAND), mskOR)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        unsafe_store!(reinterpret(Ptr{NTuple{N,VecElement{UInt64}}}, dst + i), f(res, T))
        i += 8*N
    end
    return i
end

@noinline function xoshiro_bulk_simd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{Bool}, ::Val{N}, f) where {N}
    s0, s1, s2, s3 = forkRand(rng, Val(N))
    msk = ntuple(i->VecElement(0x0101010101010101), Val(N))
    i = 0
    while i + 64*N <= len
        res = _rotl23(_plus(s0,s3))
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        for k=0:7
            tmp = _lshr(res, k)
            toWrite = _and(tmp, msk)
            unsafe_store!(reinterpret(Ptr{NTuple{N,VecElement{UInt64}}}, dst + i + k*N*8), toWrite)
        end
        i += 64*N
    end
    return i
end


function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Array{Float32}, ::SamplerTrivial{CloseOpen01{Float32}})
    GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst)*4, Float32, xoshiroWidth(), _minus1)
    dst
end

function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Array{Float64}, ::SamplerTrivial{CloseOpen01{Float64}})
    GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst)*8, Float64, xoshiroWidth(), _minus1)
    dst
end

for T in BitInteger_types
    @eval function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Union{Array{$T}, UnsafeView{$T}}, ::SamplerType{$T})
        GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst)*sizeof($T), UInt8, xoshiroWidth())
        dst
    end
end

function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Array{Bool}, ::SamplerType{Bool})
    GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst), Bool, xoshiroWidth())
    dst
end

end # module
