import Fastify from 'fastify';
import multipart from '@fastify/multipart';
import { Queue } from 'bullmq';

const fastify = Fastify({ logger: true });
fastify.register(multipart);

const imageQueue = new Queue('photo-restoration', {
  connection: { host: 'localhost', port: 6379 }
});

fastify.post('/upload', async (req, reply) => {
  const data = await req.file();
  const buffer = await data.toBuffer();

  // Envia a imagem para a fila de processamento
  const job = await imageQueue.add('restore-task', {
    filename: data.filename,
    buffer: buffer.toString('base64')
  });

  return { jobId: job.id, status: 'queued' };
});

fastify.listen({ port: 3000 }, (err) => {
  if (err) throw err;
  console.log('API Fastify ativa na porta 3000');
});