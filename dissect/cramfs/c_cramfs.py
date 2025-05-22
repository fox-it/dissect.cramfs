from dissect.cstruct import cstruct

# references
# - https://github.com/digiampietro/lzma-uncramfs/blob/master/cramfs_fs.h
# - https://github.com/npitre/cramfs-tools/blob/master/linux/cramfs_fs.h

cramfs_def = """
#define CRAMFS_MAGIC		0x28cd3d45	/* some random number */
#define CRAMFS_SIGNATURE	"Compressed ROMFS"

/*
 * Width of various bitfields in struct cramfs_inode.
 * Primarily used to generate warnings in mkcramfs.
 */
#define CRAMFS_MODE_WIDTH 16
#define CRAMFS_UID_WIDTH 16
#define CRAMFS_GID_WIDTH 8
#define CRAMFS_NAMELEN_WIDTH 6
#define CRAMFS_OFFSET_WIDTH 26

/*
 * Since inode.namelen is a unsigned 6-bit number, the maximum cramfs
 * path length is 63 << 2 = 252.
 */
#define CRAMFS_MAXPATHLEN (((1 << CRAMFS_NAMELEN_WIDTH) - 1) << 2)
#define CRAMFS_SIZE_WIDTH 24

struct cramfs_inode {
    uint32 mode:16;
    uint32 uid:16;
    /* SIZE for device files is i_rdev */

    uint32 size:24;
    uint32 gid:8;
    /* NAMELEN is the length of the file name, divided by 4 and rounded up.  (cramfs doesn't support hard links.) */
    /* OFFSET: For symlinks and non-empty regular files, this
        contains the offset (divided by 4) of the file data in
        compressed form (starting with an array of block pointers;
        see README).  For non-empty directories it is the offset
        (divided by 4) of the inode of the first file in that
        directory.  For anything else, offset is zero. */

    uint32 namelen:6;
    uint32 offset:26;
    char name[namelen * 4];
};

struct cramfs_info {
    uint32 crc;
    uint32 edition;
    uint32 blocks;
    uint32 files;
};

/*
 * Superblock information at the beginning of the FS.
 */
struct cramfs_super_block {
    uint32 magic;			/* 0x28cd3d45 - random number */
    uint32 size;			/* length in bytes */
    uint32 flags;			/* feature flags */
    uint32 future;			/* reserved for future use */
    char signature[16];		/* "Compressed ROMFS" */
    cramfs_info fsid;	    /* unique filesystem info */
    char name[16];			/* user-defined name */
    cramfs_inode root;	/* root inode data */
};

/*
 * Feature flags
 *
 * 0x00000000 - 0x000000ff: features that work for all past kernels
 * 0x00000100 - 0xffffffff: features that don't work for past kernels
 */
#define CRAMFS_FLAG_FSID_VERSION_2	    0x00000001	/* fsid version #2 */
#define CRAMFS_FLAG_SORTED_DIRS		    0x00000002	/* sorted dirs */
#define CRAMFS_FLAG_HOLES		        0x00000100	/* support for holes */
#define CRAMFS_FLAG_WRONG_SIGNATURE	    0x00000200	/* reserved */
#define CRAMFS_FLAG_SHIFTED_ROOT_OFFSET	0x00000400	/* shifted root fs */
#define CRAMFS_FLAG_BLKSZ_MASK		    0x00003800	/* Block size mask */
#define CRAMFS_FLAG_COMP_METHOD_MASK	0x0000C000	/* Compression method mask */
#define CRAMFS_FLAG_EXT_BLOCK_POINTERS	0x00000800	/* block pointer extensions */

#define CRAMFS_FLAG_BLKSZ_SHIFT	11
#define CRAMFS_FLAG_COMP_METHOD_SHIFT 14

#define CRAMFS_FLAG_COMP_METHOD_NONE 0
#define CRAMFS_FLAG_COMP_METHOD_GZIP 1
#define CRAMFS_FLAG_COMP_METHOD_LZMA 2

/*
 * Valid values in super.flags.  Currently we refuse to mount
 * if (flags & ~CRAMFS_SUPPORTED_FLAGS).  Maybe that should be
 * changed to test super.future instead.
 */
#define CRAMFS_SUPPORTED_FLAGS	( 0x000000ff | CRAMFS_FLAG_HOLES | CRAMFS_FLAG_WRONG_SIGNATURE | CRAMFS_FLAG_SHIFTED_ROOT_OFFSET | CRAMFS_FLAG_BLKSZ_MASK | CRAMFS_FLAG_COMP_METHOD_MASK)
"""  # noqa: E501

UNCOMPRESSED_BLOCK_FLAG = 0x80000000
DIRECT_POINTER_FLAG = 0x40000000
BLOCK_SIZE = 4096

c_cramfs = cstruct().load(cramfs_def)
