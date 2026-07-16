#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

static void tdxploit_marker(void)
{
    asm volatile(
        "xor %%rax, %%rax;"
        "xor %%r10, %%r10;"
        "mov $0xff00, %%rcx;"
        "mov $0x40000042, %%r12;"
        "mov $2, %%r13;"
        "mov $10, %%r11;"
        "tdcall;"
        ::: "rax", "rcx", "r10", "r11", "r12", "r13",
            "r8", "r9", "r14", "r15", "rbx", "memory"
    );
}

static ssize_t marker_write(struct file *file, const char __user *buf,
                            size_t len, loff_t *off)
{
    tdxploit_marker();
    return len;
}

static const struct file_operations marker_fops = {
    .owner = THIS_MODULE,
    .write = marker_write,
};

static struct miscdevice marker_dev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "dnn_marker",
    .fops = &marker_fops,
};

static int __init marker_init(void)
{
    return misc_register(&marker_dev);
}

static void __exit marker_exit(void)
{
    misc_deregister(&marker_dev);
}

module_init(marker_init);
module_exit(marker_exit);
MODULE_LICENSE("GPL");
