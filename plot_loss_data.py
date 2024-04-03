import matplotlib.pyplot as plt
import json

if __name__=='__main__':
    with open('loss_data.json', 'r') as f:
        content=f.read()
        content=json.loads(content)

        print(dict(content).keys())

        plt.subplot(221)
        plt.title('Generator Loss')
        plt.plot(range(0, content['loss_iterations']), json.loads(content['loss_gen_values']), 'r')
        plt.subplot(222)
        plt.title('Discriminator Loss')
        plt.plot(range(0, content['loss_iterations']), json.loads(content['loss_disc_values']), 'g')
        plt.subplot(223)
        plt.title('Per-Pixel Loss')
        plt.plot(range(0, content['loss_iterations']), json.loads(content['loss_gen_pix_values']), 'b')
        plt.subplot(224)
        plt.title('All Losses')
        plt.plot(range(0, content['loss_iterations']), json.loads(content['loss_gen_values']), 'r')
        plt.plot(range(0, content['loss_iterations']), json.loads(content['loss_disc_values']), 'g')
        plt.plot(range(0, content['loss_iterations']), json.loads(content['loss_gen_pix_values']), 'b')
        plt.savefig('loss_plot.png')
        plt.show()

