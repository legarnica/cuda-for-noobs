package clases;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.Date;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

public class Main {

    public static final int n = 10_000_000;
    public static int a[] = new int[n];

    static {
        for (int i = 2; i < n; i++) {
            a[i] = i;
        }
    }

    public static boolean isPrimo(int r) {
        int raiz = (int) Math.ceil(Math.sqrt(r));
        for (int i = 2; i <= raiz; i++) {
            if (r % i == 0) {
                return false;
            }
        }
        return true;
    }

    public static void primos() {
        int cantidad_de_primos = 0;
        for (int i = 0; i < n; i++) {
            cantidad_de_primos += isPrimo(a[i]) ? 1 : 0;
        }
        System.out.println("Hay: " + cantidad_de_primos + " primos entre 0 y " + n);
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Iniciando");
        Date now = Date.from(Instant.now());
        primos();
        Date end = Date.from(Instant.now());
        System.out.print("Sin CUDA: ");
        System.out.println((end.getTime() - now.getTime()) / 1000.0F + " segundos");
        now = Date.from(Instant.now());

        JCudaDriver.setExceptionsEnabled(true);

        String kernelFileName = prepararPTX("C:\\Users\\Carlos\\Desktop\\KernelExterno.cu");

        JCudaDriver.cuInit(0);

        //inicializar el driver y crear el contexto para el primer device
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        //cargar el ptx
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, kernelFileName);

        //obtener la funcion a usar
        CUfunction funcion = new CUfunction();
        JCudaDriver.cuModuleGetFunction(funcion, module, "primo");

        //alojar y llenar la información de input del host
        //inicializador y eso jeje
        int raices[] = new int[n];
        for (int i = 0; i < n; i++) {
            raices[i] = (int) Math.ceil(Math.sqrt(a[i]));
        }

        //alojar y llenar la info de input del device
        CUdeviceptr deviceNumero = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceNumero, n * Sizeof.INT);
        cuMemcpyHtoD(deviceNumero, Pointer.to(a), n * Sizeof.INT);
        
        CUdeviceptr deviceRaiz = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceRaiz, n * Sizeof.INT);
        cuMemcpyHtoD(deviceRaiz, Pointer.to(raices), n * Sizeof.INT);

        CUdeviceptr outPut = new CUdeviceptr();
        //JCudaDriver.cuMemAlloc(outPut, Sizeof.INT);
        JCudaDriver.cuMemAlloc(outPut, n * Sizeof.SHORT);

        //setup a los parametros del kernel
        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(deviceNumero),
                Pointer.to(deviceRaiz),
                Pointer.to(outPut)
        );

        int blockSizeX = 1024; //2.579
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        JCudaDriver.cuLaunchKernel(funcion,
                gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, null,
                kernelParams, null);
        JCudaDriver.cuCtxSynchronize();
        
        //int hostOutput[] = new int[1];
        //JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), outPut, Sizeof.INT);
        short hostOutput[] = new short[n];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), outPut, n * Sizeof.SHORT);
        
        System.out.println("Hay: " + ArraySum(hostOutput) + " primos entre 0 y " + n);

        end = Date.from(Instant.now());
        System.out.print("Con CUDA: ");
        System.out.println((end.getTime() - now.getTime()) / 1000.0F + " segundos");

    }

    private static String prepararPTX(String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command
                = "nvcc " + modelString + " -ptx "
                + cuFile.getPath() + " -o " + ptxFileName;

        System.out.println("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage
                = new String(toByteArray(process.getErrorStream()));
        String outputMessage
                = new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    private static byte[] toByteArray(InputStream inputStream)
            throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    private static int ArraySum(short[] hostOutput) {
        int r = 0;
        for (short s : hostOutput) {
            r += s;
        }
        return r;
    }
}
